import os
import glob
import sys
from copy import copy
from itertools import product
from functools import reduce
from collections import Iterable, defaultdict

import numpy as np
import pandas as pd

import scipy.stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler
import gseapy as gp

from .statistic import fdr, core_corr, core_corr_tensor, ranksums_vectorized, f_oneway_vectorized, pearson_pvalue, spearman_pvalue
from .algorithm import core_PCA, core_tSNE, core_UMAP
from .function import percentage, dropnan, handle_colors, sort_custom, ora, filter_by_quantile
from .exceptions import MethodError
from .eplot.core import scatterplot, barplot, cateplot, heatmap
from .eplot.base import savefig, confidence_ellipse
from .eplot.constants import MCMAP, R_CMAP
from .eplot.plot_func import volcanoplot


class Dataset:
    '''
    pass
    '''

    def __init__(self, dirpath=os.getcwd()):
        '''
        Initialize variables for the dataset object
        '''
        
        self._dirpath = dirpath

        if not os.path.isdir(os.path.join(dirpath, 'document')):
            FileNotFoundError("The document file is not exist")

        self.__object_name = os.path.split(os.path.abspath(self._dirpath))[-1]
        self._data = {}
        self._valid_dfs_names = []
        self._color_map = {}
        omic_names = [
            f.split('.', -1)[0] for f in os.listdir(os.path.join(self._dirpath, 'document'))]
        self._load_dataset(omic_names)
        self._initialized = True

    def _get_df_path(self, omic_name):
        '''
        Get the absolute path for a dataset
        '''
        try:
            dataset_path = glob.glob(
                os.path.join(self._dirpath, 'document', omic_name + '.*'))[0]
        except:
            raise ValueError('Please check your name parameter.')

        return dataset_path

    def _get_dataframe(self, omic_name):
        '''
        Get the specific omic dataset
        '''

        omic_path = self._get_df_path(omic_name)
        if omic_path.endswith('.maf'):
            index_col = None
        elif omic_name == 'phospho':
            index_col = [0]
        elif omic_name == 'color':
            index_col = [0, 1]
        else:
            index_col = [0]

        if omic_path.endswith('.csv'):
            df = pd.read_csv(omic_path, index_col=index_col)

        elif omic_path.endswith(('.txt', 'tsv', 'maf')):
            df = pd.read_table(omic_path, index_col=index_col)

        else:
            if omic_path.endswith('.pickle'):
                df = pd.read_pickle(omic_path)
            elif omic_path.endswith('.feather'):
                df = pd.read_feather(omic_path)
            else:
                raise ValueError(
                    "dataset file type should be one of csv, tsv, txt, pickle, feather, please check your file type."
                )
            df = df.set_index(df.columns[index_col].tolist())

        df.name = omic_name
        self._data[omic_name] = df

        if not omic_name in self._valid_dfs_names:
            self._valid_dfs_names.append(omic_name)

    def _load_dataset(self, names):
        for omic_name in names:
            try:
                self._get_dataframe(omic_name)
            except:
                continue
        if 'color' in names:
            self._color_map = handle_colors(
                self._data['color'], self._data['category'])

    def update(self, *name):
        if not name or 'all' in name:
            name = [
                f.split('.', -1)[0] for f in os.listdir(os.path.join(self._dirpath, 'document'))]
        self._load_dataset(name)

    @staticmethod
    def delete_nan_array(df, axis=1):
        if df.max(axis=axis).nunique() != 1:
            return df[df.max(axis=axis) != df.min(axis=axis)]
        else:
            return df

    def write_table(self,
                    matrix,
                    index=None,
                    columns=None,
                    index_name=None,
                    columns_name=None,
                    file_name='out',
                    out_file_type='csv',
                    **kwargs):
        if isinstance(matrix, (pd.DataFrame, pd.Series)):
            table_out = matrix
        else:
            table_out = pd.DataFrame(matrix, index=index, columns=columns).rename_axis(
                index=index_name, columns=columns_name)
        outpath = os.path.join(self._dirpath, 'document')
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        write_table_outpath = os.path.join(
            outpath, file_name + '.' + out_file_type)

        if out_file_type.endswith('pickle'):
            table_out.reset_index().to_pickle(write_table_outpath, **kwargs)
        else:
            if out_file_type.endswith('csv'):
                sep = ','
            else:
                sep = '\t'
            table_out.to_csv(write_table_outpath, sep=sep, **kwargs)

    def _handle_group(self, file_type, group_name, part_element=None):
        group_file = self._data[file_type][group_name].dropna()
        if part_element:
            if not isinstance(group_name, str) and isinstance(group_name, Iterable):
                if not isinstance(part_element[0], tuple):
                    for i, j in zip(group_name, part_element):
                        if any(np.setdiff1d(np.unique(j), np.unique(group_file[i].unique()))):
                            raise ValueError('{0} with wrong elements, please check the part_element parameter'.format(j))
                    part_element_use = list(product(*part_element))
                group_file = group_file.loc[np.isin(group_file.values, part_element_use).all(axis=1)].pipe(sort_custom, group_name, part_element)
            else:
                group_file = group_file.loc[np.isin(group_file.values, part_element)].pipe(sort_custom, group_name, part_element)
        return group_file

    def __getattr__(self, __name):
        if __name in self._data:
            return self._data[__name]
        else:
            return object.__getattribute__(self, __name)

    def __str__(self):
        return 'Load {0} datasets from {1} project:\n {2}'.format(
            len(self._valid_dfs_names), self.__object_name, '\n '.join(self._valid_dfs_names))


class Preprocessing(Dataset):
    def __init__(self,
                 dirpath=os.getcwd(),
                 ) -> None:
        super().__init__(dirpath)
    
    def scale(self, element, scaler):
        if not scaler:
            return element
        elif isinstance(element, str):
            df = self._data[element]
        elif isinstance(element, pd.Series):
            df = element.to_frame()
        else: 
            df = element

        scaler_methods = {'standard': StandardScaler, 'zscore': StandardScaler, 'minmax': MinMaxScaler, 'normalizer': Normalizer, 'robust': RobustScaler, 'log2': np.log2, 'log10': np.log10}
        scaler = scaler_methods[scaler]
        
        if scaler in [np.log2, np.log10]:
            return scaler(df)
        else:
            return pd.DataFrame(scaler().fit_transform(df), index=df.index, columns=df.columns)

    def calculate_vif(self, df, thresh=5):
        drop = True
        df = df.assign(add_const=1)
        while drop:
            variables = df.columns
            vif = np.array([variance_inflation_factor(df, i) for i in range(df.shape[1])])[:-1]
            max_vif = np.max(vif)
            if max_vif > thresh:
                max_loc = np.argmax(max_vif)
                print(f"Dropping {df.columns[max_loc]} with vif={max_vif}")
                df = df.drop(variables[max_loc], axis=1)
            else:
                drop = False
        return df.drop('add_const', axis=1)

class Group(Preprocessing):
    def __init__(self,
                 dirpath=os.getcwd(),
                 group_name=None,
                 dataset_type=None,
                 file_type='category',
                 thresh=1e-5,
                 part_element=None,
                 param_method='mean',
                 statistic_method='log2',
                 ttest_kwargs = {},
                 fdr_method='i',
                 dividend=None,
                 palette=None,
                 *args,
                 **kwargs):

        super().__init__(dirpath)
        self._group_name = group_name
        self._file_type = file_type
        self._dataset_type = dataset_type
        self._part_element = part_element
        self._thresh = thresh
        self._param_method = param_method
        self._statistic_method = statistic_method
        self._ttest_kwargs = ttest_kwargs
        self._fdr_method = fdr_method
        self._dividend = dividend
        self._palette = palette
        
        self._group_check_params()

    def _group_check_params(self):
        if self._group_name and self._file_type and self._dataset_type:
            if any(np.setdiff1d(self._group_name, self._data[self._file_type].columns)):
                raise ValueError(
                    "{0} is not in {1} dataset, please check the group_name parameter."
                    .format(self._group_name, self._file_type))
            self.__group_pipeline()
        elif self._dataset_type:
            self._tmp_dataset = self._data[self._dataset_type]

    def __group_pipeline(self):
        self._part_element_keep_for_palette = copy(self._part_element)
        self._group_file = self._handle_group(self._file_type, self._group_name, self._part_element)
        
        if isinstance(self._group_file, pd.Series):
            self._group_file = self._group_file.to_frame()

        self._tmp_dataset = self._data[self._dataset_type].reindex(
            self._group_file.index,
            axis=1).dropna(axis=1, how='all').pipe(self.delete_nan_array).pipe(dropnan, thresh=self._thresh)
        self._group_file = self._group_file.loc[self._tmp_dataset.columns].apply(lambda x: '_'.join(x.astype(str)), axis=1)
        # if self._part_element and not isinstance(self._group_name, str) and not isinstance(self._part_element[0], tuple):
        self._part_element = pd.unique(self._group_file).tolist()

        tmp = {
            name: self._tmp_dataset.reindex(group.index,
                                            axis=1).dropna(axis=1, how='all').values
            for name, group in self._group_file.groupby(self._group_file, sort=False)
        }

        self._group_values = list(tmp.keys())
        self.__group = list(tmp.values())
        self.__group_set_params()
        
        self.__group_cal_values()
        if len(self._part_element) != 1:
            self.__table()

    def __group_set_params(self):
        if len(self._group_values) == 2:
            if self._dividend == self._group_values[0]:
                self._change = False
                self._dividend, self._divisor = self._group_values

            else:
                self._change = True
                self._divisor, self._dividend = self._group_values
        if not self._palette:
            if isinstance(self._group_name, list):
                for_palette_dict = dict(zip(self._group_name, self._part_element_keep_for_palette))
                palette_from_file_count = 0
                palette_from_file = True
                for k, v in for_palette_dict.items():
                    if len(v) != 1:
                        palette_from_file_count += 1
                        pivot_group_name = k
                        if palette_from_file_count > 1:
                            palette_from_file = False
            else:
                palette_from_file = isinstance(self._group_name, str)
                pivot_group_name = self._group_name
                palette_from_file_count = 1
            if palette_from_file and palette_from_file_count == 1:
                palette = self._color_map.get(pivot_group_name, MCMAP[:len(self._group_values)])
                if isinstance(palette, list):
                    palette = dict(zip(self._part_element, palette))
                self._palette = {}
                for i in self._part_element:
                    for k, v in palette.items():
                        if k == i:
                            self._palette[i] = v
                        elif k in i:
                            if k in self._palette.keys():
                                continue
                            else:
                                self._palette[i] = v             
                if self._part_element:
                    self._palette = {k: v for k, v in self._palette.items() if k in self._part_element}
                else:
                    self._part_element = pivot_group_name
            else:
                self._palette = {k: MCMAP[i] for i, k in enumerate(self._part_element)}

    def __group_cal_values(self):
        mean_value = np.array(
            list(map(lambda x: np.nanmean(x, axis=1), self.__group)))
        median_value = np.array(
            list(map(lambda x: np.nanmedian(x, axis=1), self.__group)))
        std_value = np.array(
            list(map(lambda x: np.nanstd(x, axis=1, ddof=1), self.__group)))
        count_value = np.array(
            list(map(lambda x: np.count_nonzero(~np.isnan(x), axis=1), self._Group__group)))
        percentage_value = np.array(
            list(map(lambda x: percentage(x, axis=1), self.__group)))
        cv_value = std_value / mean_value

        if len(self._group_values) == 1:
            inference_statistics = []

        elif len(self._group_values) == 2:
            inference_statistics = self.__two_groups_cal_statistic_prob(count=count_value)
        else:
            inference_statistics = self.__multi_groups_cal_statistic_prob()

        self.__param_values = dict(
            zip(['mean', 'median', 'std', 'cv', 'count', 'percentage', 'inference_statistics'], [
                mean_value, median_value, std_value, cv_value, count_value, 
                percentage_value, inference_statistics
            ]))

        self.__out_index = np.any(
            self.__param_values['percentage'] > self._thresh, axis=0)

    def __dateset_preprocess_for_statistic(self):
        statistic_methods = {'log2': np.log2, 'log10': np.log10}

        if self._statistic_method in statistic_methods.keys(
        ) and self._tmp_dataset.min().min() > 0:
            statistic_data = list(
                map(
                    lambda x: statistic_methods[self._statistic_method]
                    (x), self.__group))
        else:
            statistic_data = self.__group

        return statistic_data

    def __two_groups_cal_statistic_prob(self, count=None):
        statistic_data = self.__dateset_preprocess_for_statistic()

        ttest_statistic, ttest_pvalues = np.asarray(scipy.stats.ttest_ind(*statistic_data, axis=1, equal_var=True, nan_policy='omit', **self._ttest_kwargs))
        # ttest_pvalues = np.asarray(scipy.stats.ttest_ind(*statistic_data, axis=1, equal_var=True, nan_policy='omit')[1])
        adjust_ttest_pvalues = fdr(ttest_pvalues, self._fdr_method)[1]
        ranksums_statistic, ranksums_pvalues = ranksums_vectorized(statistic_data[0], statistic_data[1])
        adjust_ranksums_pvalues = fdr(ranksums_pvalues.copy(), self._fdr_method)[1]
        if isinstance(count, np.ndarray):
            cohen_d = ttest_statistic * np.sqrt(1 / count[0] + 1 / count[1])
            wilcoxon_es = ranksums_statistic / np.sqrt(count.sum(axis=0))
        else:
            cohen_d = None
            wilcoxon_es = None
        return ttest_statistic, cohen_d, ttest_pvalues, adjust_ttest_pvalues, ranksums_statistic, wilcoxon_es, ranksums_pvalues, adjust_ranksums_pvalues

    def __multi_groups_cal_statistic_prob(self):
        statistic_data = self.__dateset_preprocess_for_statistic()
        # anova_statistics, anova_pvalues = f_oneway_vectorized(*statistic_data, axis=1)
        anova_statistics, anova_pvalues = scipy.stats.f_oneway(*statistic_data, axis=1, nan_policy='omit')
        adjust_anova_pvalues = fdr(anova_pvalues, self._fdr_method)[1]
        res = scipy.stats.alexandergovern(*statistic_data, axis=1, nan_policy='omit')
        alexandergovern_statistics = res.statistic
        alexandergovern_pvalues = res.pvalue
        adjust_alexandergovern_pvalues = fdr(alexandergovern_pvalues, self._fdr_method)[1]
        kruskal_statistics, kruskal_pvalues = np.asarray(scipy.stats.kruskal(*statistic_data, axis=1, nan_policy='omit'))
        adjust_kruskal_pvalues = fdr(kruskal_pvalues, self._fdr_method)[1]

        return anova_statistics, anova_pvalues, adjust_anova_pvalues, alexandergovern_statistics, alexandergovern_pvalues, adjust_alexandergovern_pvalues, kruskal_statistics, kruskal_pvalues, adjust_kruskal_pvalues

    def __table(self):
        tmp_param_values = self.__param_values[self._param_method]
        group_values = [str(i) for i in self._group_values]
        if len(group_values) == 2:
            if any(np.hstack(tmp_param_values)<0):
                ratio = tmp_param_values[0] - tmp_param_values[1]
                nega_annot = True 
            else:
                ratio = tmp_param_values[0] / tmp_param_values[1]
                nega_annot = False 
            
            ratio_label = self._dividend + '_vs_' + self._divisor
            table_columns = [
                group_values[0] + '_' + self._param_method,
                group_values[1] + '_' + self._param_method,
                ratio_label,
                'ttest_statistics',
                'cohen_d',
                'ttest_pvalues',
                'ttest_fdr',
                'ranksums_statistics',
                'wilcoxon_es',
                'ranksums_pvalues',
                'ranksums_fdr',
                group_values[0] + '_percentage',
                group_values[1] + '_percentage',
            ]
            self._group_table = pd.DataFrame(np.vstack(
                (tmp_param_values, ratio,
                 self.__param_values['inference_statistics'],
                 self.__param_values['percentage'])).transpose(),
                index=self._tmp_dataset.index,
                columns=table_columns)
            if self._change:
                if nega_annot:
                    self._group_table.iloc[:, 2] = -self._group_table.iloc[:, 2]
                else:
                    self._group_table.iloc[:, 2] = 1.0 / self._group_table.iloc[:, 2]
                # self._group_table.name = '_'.join(
                #     [self._dataset_type, self._group_name])

        else:
            table_columns = [
                '_'.join((label, self._param_method))
                for label in group_values
            ]
            table_columns.extend(
                ['anova_statistics', 'anova_pvalues', 'anova_fdr', 'alexandergovern_statistics', 'alexandergovern_pvalues', 'alexandergovern_fdr', 'kruskal_statistics', 'kruskal_pvalues', 'kruskal_fdr'])
            table_columns.extend([
                '_'.join((label, 'percentage'))
                for label in group_values
            ])
            self._group_table = pd.DataFrame(np.vstack(
                (tmp_param_values, self.__param_values['inference_statistics'],
                 self.__param_values['percentage'])).transpose(),
                index=self._tmp_dataset.index,
                columns=table_columns)

    @property
    def table(self):
        return self._group_table.loc[self._tmp_dataset.index[self.__out_index]]

    @property
    def param_table(self):
        param_table_columns = []
        for name in ['mean', 'median', 'standard', 'cv', 'count', 'percentage']:
            param_table_columns.extend([
                group_name + '_' + name for group_name in self._group_values
            ])

        self._param_table = pd.DataFrame(np.vstack(
            (self.__param_values['mean'], self.__param_values['median'], self.__param_values['std'], self.__param_values['cv'],
             self.__param_values['count'], self.__param_values['percentage'])).transpose(),
            index=self._tmp_dataset.index,
            columns=param_table_columns)
        return self._param_table

    def merge_data_group(self, data_element, group_name=None, data_type=None, part_element=None, join_method='inner', sort='element', sort_group=None, ascending=True, shuffle=False):
        if not group_name:
            group_name = self._group_name
        if isinstance(group_name, str):
            group_name = [group_name]
            
        group_name_df = self._data[self._file_type].loc[:, group_name].dropna(how='all').loc[self._group_file.index]
        if not part_element:
            part_element = self._part_element

        # group_name_df = group_name_df[group_name_df[self._group_name].fillna(' ').apply(lambda x: '_'.join(x), axis=1).isin(part_element)].dropna().rename_axis('sample')
        # group_name_df = sort_custom(group_name_df, order=part_element, label=self._group_name)

        columns_name = self._tmp_dataset.index.name
        if data_type:
            data = pd.DataFrame(columns=group_name_df.index)
            for dt in data_type:
                add_df = self._data[dt].reindex(self._tmp_dataset.columns, axis=1).dropna(how='all', axis=1).reindex(data_element).dropna(how='all')
                add_df.index = add_df.index + '|' + dt
                data = pd.concat([data, add_df], join=join_method)
            if sort.startswith('el'):
                data.index = pd.MultiIndex.from_tuples(data.index.str.split('|', n=-1).tolist())
                data = data.loc[data_element]
                data.index = data.index.map('|'.join)
        else:
            data = self._tmp_dataset.reindex(data_element).dropna(how='all')

        data = pd.concat([data.T, group_name_df], axis=1, join='inner').set_index(group_name, append=True)
        if shuffle:
            data = data.groupby(level=group_name, as_index=False, sort=False).apply(lambda x: x.sample(x.shape[0]))
        # print(data.head())
        # if not sort_group:
        #     data = data.sort_index(axis=0, level=group_name, ascending=ascending)   # 没有解决何种排序的问题， ABC or CBA or CAB ...
        # else:
        #     data = data.sort_values(by=sort_group, axis=0, ascending=ascending)
        data.columns.name = columns_name
        return data

    def decomposition(self,
                      method='pca',
                      transform=None,
                      components=[1, 2],
                      dec_kwargs={},
                      scatter_kwargs={},
                      nsd=None,
                      ellipse_kwargs={},
                      annot_outlier=False,
                      return_pca=False,
                      ticklabels_hide=[],
                      labels_hide=[],
                      ticklabels_format=[]):
        df = self._tmp_dataset.loc[self.__out_index].fillna(1e-5)

        if transform and transform != 'no':
            try:
                if 'log' in transform and df.min().min() < 1:
                    df = df + 1
                transform_methods = {'log2': np.log2, 'log10': np.log10, 'minmax': MinMaxScaler().fit_transform, 'standard': StandardScaler(
                ).fit_transform, 'normalizer': Normalizer().fit_transform, 'robust': RobustScaler().fit_transform}.get(transform)
                df_scaler = transform_methods(df.T)
            except:
                raise ValueError(
                    'transform parameter should be one of the log2, log10, minmax, standard, normalizer, and robust.')
        else:
            df_scaler = df.T

        if self._group_name:
            label = self._data[self._file_type][self._group_name].loc[df.columns].values
            columns = ['label']
        else:
            columns = []

        if method.lower() == 'pca':
            decomposition_vector, evr = core_PCA(df_scaler, **dec_kwargs)
            columns.extend(["PCA{} ({:.2%})".format(i, evr[i-1]) for i in components])

        elif method.lower() == 'tsne':
            decomposition_vector = core_tSNE(df_scaler, **dec_kwargs)
            columns.extend(['tSNE {}'.format(i) for i in components])

        elif method.lower() == 'umap':
            decomposition_vector = core_UMAP(df_scaler, **dec_kwargs)
            columns.extend(['UMAP {}'.format(i) for i in components])


        if self._group_name:
            if not isinstance(self._group_name, str):
                label = self._group_file.values
                title_part_name = '-'.join(self._group_name)
            else:
                title_part_name = self._group_name

            plotdata = pd.DataFrame(np.hstack(
                [label[:, None], decomposition_vector[:, np.asarray(components)-1]]), columns=columns, index=df.columns)
        else:
            plotdata = pd.DataFrame(
                decomposition_vector[:, np.asarray(components)-1], columns=columns, index=df.columns)
        
        if len(components) == 2:
            plotdata = plotdata.iloc[:, [1, 0, 2]]
            ax = scatterplot(plotdata, title={'pca': 'PCA', 'tsne': 'tSNE', 'umap': 'UMAP'}.get(method.lower()) + ' of {}'.format(title_part_name), ticklabels_hide=ticklabels_hide, labels_hide=labels_hide, ticklabels_format=ticklabels_format, palette=self._palette, **scatter_kwargs)

            if nsd:
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
                for i in self._group_values:
                    tmp = plotdata.query("label==@i")
                    x, y = tmp.iloc[:, 0].astype(float).values, tmp.iloc[:, 2].astype(float).values

                    xlim1, ylim1 = confidence_ellipse(x, y, ax, edgecolor=self._palette[i], n_std=nsd, facecolor=self._palette[i], alpha=.2, lw=2, **ellipse_kwargs)
                    # if annot_outlier:
                    #     annot_df = tmp[(tmp.iloc[:, 0]>xlim1[1])|(tmp.iloc[:, 0]<xlim1[0])|(tmp.iloc[:, 2]>ylim1[1])|(plotdata.iloc[:, 2]<ylim1[0])]
                    #     adjusttext(annot_df, annot_df.index, annot_df.columns[0], annot_df.columns[2])
                    xlim = (min(xlim[0], xlim1[0]), max(xlim[1], xlim1[1]))
                    ylim = (min(ylim[0], ylim1[0]), max(ylim[1], ylim1[1]))
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            if return_pca:
                return plotdata, ax
            else:
                return ax


class Correlation(Preprocessing):
    def __init__(self,
                 dirpath=os.getcwd(),
                 name1=None,
                 name2=None,
                 element1=None,
                 element2=None,
                 file_type='category',
                 group_name=None,
                 part_element=None,
                 thresh=0,
                 cal_type='other',
                 fdr_method='i',
                 fdr_type='local',
                 algorithm='all',
                 write_out=False,
                 *args,
                 **kwargs):
        super().__init__(dirpath)

        self._corr_name1 = name1
        self._corr_name2 = name2
        self._corr_element1 = element1
        self._corr_element2 = element2
        self._corr_file_type = file_type
        self._corr_group_name = group_name
        self._corr_part_element = part_element
        self._corr_thresh = thresh
        self._corr_cal_type = cal_type
        self._corr_fdr_method = fdr_method
        self._corr_fdr_type = fdr_type
        self._corr_algorithm = algorithm
        self._write_out = write_out

        self._corr_check_params()

    def _corr_check_params(self):
        if self._corr_name1 and self._corr_name2 and self._corr_element1 and self._corr_element2:
            self.__corr_pipeline()

    def __handle_omic_label(self, omic_label, corr_group_name):
        if isinstance(omic_label, str):
            if omic_label == 'all':
                return self._data[corr_group_name].index
            else:
                return [omic_label]
        elif isinstance(omic_label, Iterable):
            return omic_label
        else:
            raise ValueError('Please check your omic_label of {}'.format(corr_group_name))

    def __corr_func(self, algorithm, matrix1_value, matrix2_value):
        if algorithm == 'spearman':
            corr_name = algorithm + '_rho'
        else:
            corr_name = algorithm + '_corr'

        if all((len(self._handle_element1)>1, len(self._handle_element2)>1, not self._corr_cal_type.startswith('co'))):
            matrix1_value[matrix1_value!=matrix1_value] = 0
            matrix2_value[matrix2_value!=matrix2_value] = 0
            dof, corr = core_corr_tensor(matrix1_value, matrix2_value, method=algorithm)
        
        else:
            count, dof, corr = core_corr(matrix1_value, matrix2_value, method=algorithm)
            self._corr_value['count'] = self._corr_value.get('count', count)
            if self._corr_cal_type.startswith('co'):
                self._corr_value['frequence_'+self._corr_name1] = self._corr_value.get('frequence_'+self._corr_name1, count / np.count_nonzero(~np.isnan(matrix1_value), axis=1))
                self._corr_value['frequence_'+self._corr_name2] = self._corr_value.get('frequence_'+self._corr_name2, count / np.count_nonzero(~np.isnan(matrix2_value), axis=1))
            else:
                self._corr_value['frequence'] = self._corr_value.get('frequence', count / np.count_nonzero(~np.isnan(matrix1_value), axis=1))
            self._corr_index = self._corr_value['frequence'] > self._corr_thresh

        self._corr_value[corr_name] = corr

        if algorithm == 'spearman':
            prob = spearman_pvalue(corr, dof)
        else:
            prob = pearson_pvalue(corr, dof)

        self._corr_value[algorithm+'_pvalues'] = prob
        if self._corr_fdr_type == 'global':
            prob = prob.faltten()
        if prob.ndim == 1:
            prob = prob[None, :]
        # self._corr_value['count'] = self._corr_value.get('count', count)
        fdr_ = np.apply_along_axis(
            fdr, 1, prob, method=self._corr_fdr_method)[:, 1][0]   # non corresponding FDR not solved
        self._corr_value[algorithm+'_fdr'] = fdr_

    def __cal_cor_value(self, matrix1_value, matrix2_value):
        if 'spearman' in self._corr_algorithm.lower() or 'all' == self._corr_algorithm.lower():  # 输入参数时做'all'替换
            self.__corr_func('spearman', matrix1_value, matrix2_value)
        if 'pearson' in self._corr_algorithm.lower() or 'all' == self._corr_algorithm.lower():
            self.__corr_func('pearson', matrix1_value, matrix2_value)

    def __corr_pipeline(self):
        if self._corr_group_name and self._corr_file_type:
            omic_group_name_id = self._handle_group(self._corr_file_type, self._corr_group_name, self._corr_part_element).index
        else:
            omic_group_name_id = self._data[self._corr_file_type].index

        pre_element1 = self.__handle_omic_label(
            self._corr_element1, self._corr_name1)
        pre_element2 = self.__handle_omic_label(
            self._corr_element2, self._corr_name2)

        if min(len(pre_element1), len(pre_element2)) < 1:
            raise ValueError('Please check the element1 or the element2')

        if len(pre_element2) < len(pre_element1):
            pre_element1, pre_element2 = pre_element2, pre_element1
            self._corr_element1,  self._corr_element2 =  self._corr_element2,  self._corr_element1
            self._corr_name1, self._corr_name2 = self._corr_name2, self._corr_name1
        
        # for 1 vs. multi
        if min(len(pre_element1), len(pre_element2)) == 1:
            pre_omic1_dataset = self._data[self._corr_name1].loc[pre_element1].dropna(how='all', axis=1)
            self._corr_columns = np.intersect1d(omic_group_name_id, pre_omic1_dataset.columns)
            pre_omic2_dataset = self.delete_nan_array(self._data[self._corr_name2].loc[pre_element2].reindex(self._corr_columns, axis=1).dropna(how='all', axis=1)).pipe(dropnan, thresh=self._corr_thresh, min_num=False)
            self._corr_columns = np.intersect1d(self._corr_columns, pre_omic2_dataset.columns)
            pre_omic1_dataset = pre_omic1_dataset.loc[:, self._corr_columns]
        # for corresponding correlation
        else:
            self._corr_columns = reduce(np.intersect1d, (self._data[self._corr_name1].columns, self._data[self._corr_name2].columns, omic_group_name_id))
            pre_omic1_dataset = self.delete_nan_array(self._data[self._corr_name1].loc[pre_element1, self._corr_columns]).pipe(dropnan, thresh=self._corr_thresh, min_num=False)
            pre_omic2_dataset = self.delete_nan_array(self._data[self._corr_name2].loc[pre_element2, self._corr_columns]).pipe(dropnan, thresh=self._corr_thresh, min_num=False)

            if self._corr_cal_type.startswith('co'):
                self._handle_element1 = self._handle_element2 = pre_omic1_dataset.index.intersection(pre_omic2_dataset.index)
                if len(self._handle_element1) == 0:
                    raise ValueError('{} and {} has not overlapped index'.format(
                        self._corr_name1, self._corr_name2))
                
                pre_omic1_dataset = pre_omic1_dataset.loc[self._handle_element1]
                pre_omic2_dataset = pre_omic2_dataset.loc[self._handle_element2]

        self._handle_element1 = pre_omic1_dataset.index
        self._handle_element2 = pre_omic2_dataset.index
        self._corr_value = {}
        self.__cal_cor_value(pre_omic1_dataset.values, pre_omic2_dataset.values)
        self.__corr_table()

        if self._write_out:
            for name, matrix in self._corr_value.items():
                self.write_table(matrix,
                                 index=self._handle_element2,
                                 columns=self._handle_element2,
                                 index_name=self._corr_name1,
                                 columns_name=self._corr_name2,
                                 file_name=self._corr_name1 + '_' +
                                 self._corr_name2 + '_' + name,
                                 out_file_type=self._write_out)

    def __corr_table(self):
        if not ((len(self._handle_element1) == 1
                 or len(self._handle_element2) == 1)
                or self._corr_fdr_type.lower() == 'local'):
            raise MethodError(
                'corr_table only suitable for one vs. n data type or n vs. n data when fdr_type is local , please consider spearman_rho, spearman_prob, spearman_fdr, pearson_corr, pearson_prob and pearsonfdr function to obtain correlation, probability and FDR matrix seperately.'
            )
        else:
            if len(self._handle_element2) >= len(self._handle_element1):
                table_index = self._handle_element2
            else:
                table_index = self._handle_element1
            self._corr_table = pd.DataFrame(np.vstack(
                list(self._corr_value.values())).T,
                index=table_index,
                columns=self._corr_value.keys())
            return self._corr_table
    
    @property
    def corr_table(self):
        return self._corr_table


class Regression(Preprocessing):
    def __init__(self, dirpath=os.getcwd(), type='ols', scaler={}, y=None, x=None, cutoff=None, y_order=None, file_type='category', group_name=None, part_element=None, thresh=None, const=True, categorical='auto', anova=False, output=['params', 'pvalues']):
        super().__init__(dirpath)
        self._reg_type = type
        self._reg_y_order = y_order
        self._reg_x = x
        self._reg_y = y
        self._reg_scaler = scaler
        self._reg_vif_cutoff = cutoff
        self._reg_const = const
        self._reg_categorical = categorical
        self._reg_anova = anova
        self._reg_file_type = file_type
        self._reg_group_name = group_name
        self._reg_part_element = part_element
        self._reg_thresh = thresh
        self._reg_output = output
        self.reg_model = defaultdict(dict)
        self._reg_check_params()

    def _reg_check_params(self):
        if self._reg_type and self._reg_x and self._reg_y:
            if not isinstance(self._reg_x, dict):
                raise TypeError("The input x should be a dictionary, please check your parameter.")
            else:
                self.__reg_pipeline()

    def __reg_pipeline(self):
        group_file = None
        if self._reg_file_type and self._reg_group_name:
            group_file = self._handle_group(self._reg_file_type, self._reg_group_name, self._reg_part_element)
        self.__reg_set_params(group_file)
        
        if isinstance(self._reg_scaler, str):
            scaler_x = self._reg_scaler
            scaler_y = self._reg_scaler
        elif isinstance(self._reg_scaler, dict):
            scaler_x = self._reg_scaler.get('x', None)
            scaler_y = self._reg_scaler.get('y', None)
        self._reg_x_df = self._reg_x_df.pipe(self.scale, scaler_x)
        if self._reg_vif_cutoff:
            self._reg_x_df = self._reg_x_df.pipe(self.calculate_vif, thresh=self._reg_vif_cutoff)
        self._reg_y_df = self.scale(self._reg_y_df, scaler_y)

        for i in self._reg_y_df.columns:
            self._reg_df = pd.concat([self._reg_y_df[i].astype(float), self._reg_x_df], axis=1, join='inner').dropna()
            try:
                model = self.__reg_fit(i)
                self.reg_model[i]['model'] = model
            except:
                print(i)
            
            if self._reg_anova:
                try:
                    self.reg_model[i]['anova'] = anova_lm(model, typ=2)
                except:
                    print('Error in {} variable'.format(i))
        self.__reg_table()

    def __reg_set_params(self, group_file):
        def create_df_from_dict(dic, filter_index=None):
            tmp = pd.DataFrame()
            for k, v in dic.items():
                tmp_data = self._data[k]
                if k in ['category'] or k.startswith('cate'):
                    tmp_data = tmp_data.T
                if v == 'all':
                    combine = tmp_data
                elif isinstance(v, str):
                    combine = tmp_data.loc[[v]]
                else:
                    combine = tmp_data.loc[v]
                tmp = pd.concat([tmp, combine.T], axis=1)
            if filter_index:
                return tmp.reindex(filter_index).dropna()
            else:
                return tmp
        if isinstance(group_file, (pd.DataFrame, pd.Series)):
            self._reg_x_df = create_df_from_dict(self._reg_x, group_file.index.tolist())
        else:
            self._reg_x_df = create_df_from_dict(self._reg_x)
        self._reg_y_df = create_df_from_dict(self._reg_y)
        if self._reg_thresh:
            self._reg_y_df = dropnan(self._reg_y_df, thresh=self._reg_thresh, axis=1)

        if self._reg_type == 'logit':
            self._reg_y_df = pd.get_dummies(self._reg_y_df).iloc[:, range(0, self._reg_y_df.shape[1]*2, 2)]
            self._regressor = smf.logit
        elif self._reg_type == 'ols':
            self._reg_y_df = self._reg_y_df.astype(float)
            self._regressor = smf.ols
        elif self._reg_type == 'softmax':
            self._regressor = smf.mnlogit

    def __reg_fit(self, y):
        formula = "{} ~ {}".format(y, ' + '.join(self._reg_x_df.columns))
        if not self._reg_const:
            formula = formula + '-1'
        if self._reg_categorical:
            if self._reg_categorical=='auto':
                count = self._reg_x_df.apply(lambda x: x.nunique())
                categorical_columns = count[count<8].index
            else:
                categorical_columns = self._reg_categorical

            for categorical_column in categorical_columns:
                formula = formula.replace(' ' + categorical_column, ' C({}) '.format(categorical_column))
        model = self._regressor(formula, data=self._reg_df).fit()
        return model

    def __reg_table(self):
        self._reg_table = pd.DataFrame()
        outname_mapping = {'params': 'coef'}
        for output in self._reg_output:
            out_dict = {}
            for name, model in self.reg_model.items():
                if output == 'eta':
                    try:
                        tmp = model['anova']
                        tmp['F'] = tmp['F']/tmp['F'].sum()
                        out_dict[name] = tmp['F'].iloc[:-1].to_dict()
                    except:
                        pass
                else:
                    out_dict[name] = getattr(model['model'], output)
            if not isinstance(list(out_dict.values())[0], (pd.Series, pd.DataFrame, dict)):
                columns = ['model']
            else:
                columns = None
            self._reg_table = pd.concat([self._reg_table, pd.DataFrame().from_dict(out_dict, orient='index', columns=columns).rename(columns=lambda x: x+'_{}'.format(outname_mapping.get(output, output)))], axis=1)

    @property
    def reg_table(self):
        return self._reg_table


class Analysis(Group, Correlation, Regression):
    def __init__(self,
                 dirpath=os.getcwd(),
                 group_name=None,
                 dataset_type=None,
                 file_type='category',
                 thresh=1e-5,
                 part_element=None,
                 param_method='mean',
                 statistic_method='log2',
                 ttest_kwargs = {},
                 fdr_method='i',
                 dividend=None,
                 palette={},
                 name1=None,
                 name2=None,
                 element1=None,
                 element2='all',
                 cal_type='other',   # corresponding, other
                 fdr_type='local',   # local / global
                 algorithm='all',   # spearman / pearson / all
                 write_out=False,
                 type='ols',
                 scaler=None,
                 y=None,
                 x=None,
                 cutoff=None,
                 y_order=None,
                 const=True,
                 categorical='auto',
                 anova=False,
                 output=['params', 'pvalues']
    ):

        super(Analysis,
              self).__init__(dirpath, group_name, dataset_type, file_type, thresh, part_element, param_method, statistic_method, ttest_kwargs, fdr_method, dividend, palette, name1, name2, element1, element2, cal_type, fdr_type, algorithm, write_out, type, scaler, y, x, cutoff, y_order, thresh, const, categorical, anova, output)

        self._corr_param_collections = {
            'name1': self._corr_name1,
            'name2': self._corr_name2,
            'element1': self._corr_element1,
            'element2': self._corr_element2,
            'cal_type': self._corr_cal_type,
            'fdr_method': self._corr_fdr_method,
            'fdr_type': self._corr_fdr_type,
            'algorithm': self._corr_algorithm,
            'file_type': self._corr_file_type,
            'group_name': self._corr_group_name,
            'part_element': self._corr_part_element,
            'thresh': self._corr_thresh,
            'write_out': self._write_out  # 未解决
        }
        self._group_param_collections = {
            'group_name': self._group_name,
            'dataset_type': self._dataset_type,
            'file_type': self._file_type,
            'thresh': self._thresh,
            'part_element': self._part_element,
            'param_method': self._param_method,
            'statistic_method': self._statistic_method,
            'ttest_kwargs': self._ttest_kwargs,
            'fdr_method': self._fdr_method,
            'dividend': self._dividend,
            'palette': self._palette
        }
        self._reg_param_collections = {
            'type': self._reg_type,
            'scaler': self._reg_scaler,
            'y_order': self._reg_y_order,
            'x': self._reg_x,
            'y': self._reg_y,
            'cutoff': self._reg_vif_cutoff,
            'const': self._reg_const,
            'categorical': self._reg_categorical,
            'anova': self._reg_anova,
            'file_type': self._reg_file_type,
            'group_name': self._reg_group_name,
            'part_element': self._reg_part_element,
            'thresh': self._reg_thresh,
            'output':self._reg_output
        }

        self._tmp_dict = {
            'group': self._group_param_collections,
            'corr': self._corr_param_collections,
            'reg': self._reg_param_collections
        }
        self._enrich_tool = {}

    def __set_default_params(self, params=[]):
        for param in params:
            setattr(self, '_'+param, None)

    def set_param(self, function_name, **kwargs):
        if function_name.lower() not in self._tmp_dict.keys():
            raise ValueError(
                'The first param of set_param function should be one of {0}, please check it.'
                .format(', '.join(self._tmp_dict.keys())))

        else:
            error_param = list(
                set(kwargs.keys()) - set(self._tmp_dict[function_name].keys())
            )
            if error_param:
                raise ValueError(
                    'Please check the input parameter name: {0}'.format(
                        ', '.join(error_param)))
            if function_name == 'group':
                params = []
                if 'group_name' in kwargs and 'group_name' != self._group_name:
                    params.extend(['dividend', 'part_element', 'palette'])
                if 'file_type' in kwargs and 'file_type' != self._file_type:
                    params.extend('group_name')
                self.__set_default_params(params)

                for k, v in kwargs.items():
                    if k == 'thresh':
                        v = {0: 1e-5, 1: 1-1e-5}.get(v, v)

                    self.__dict__['_' + k] = v
                    self._tmp_dict[function_name][k] = v
                self._group_check_params()

            elif function_name == 'corr':
                self._corr_value = {}
                for k, v in kwargs.items():
                    self.__dict__['_corr_' + k] = v
                    self._tmp_dict[function_name][k] = v
                self._corr_check_params()

            elif function_name == 'reg':
                if len(kwargs) == 1 and list(kwargs.keys())[0] == 'output':
                    self.__dict__['_reg_output'] = kwargs['output']
                    self._Regression__reg_table()
                else:
                    self.reg_model = defaultdict(dict)
                    for k, v in kwargs.items():
                        self.__dict__['_reg_' + k] = v
                    self._reg_check_params()

    def get_param(self):
        return pd.DataFrame().from_dict(self._tmp_dict).rename(columns={
            0: 'Value'
        }).rename_axis('Parameter', axis='index')

    def get_gene_from_enrichment(self, tool_term):
        enrich_name, elements = list(tool_term.items())[0]
        table_name = self._enrich_tool[enrich_name]
        term_name, gene_column_name, sep = {'clusterprofiler': ['Description', 'geneID', '/'], 'gprofiler': ['name', 'intersections', None], 'gsea_': ['Term', 'Lead_genes', ';']}.get(table_name)
        tmp = getattr(self, enrich_name).set_index(term_name).loc[elements, gene_column_name]
        if sep:
            genes = tmp.str.split(sep).to_dict()
        else:
            genes = tmp.to_dict()
        return genes

    def enrich(self,
                table_type,
                query_condition,
                tool='gprofiler',
                organism='hsapiens',
                db=None,
                bg=None,
                **kwargs):

        if table_type == 'group':
            table = self._group_table.copy()
        elif table_type == 'corr':
            table = self._corr_table.copy()
        elif table_type == 'reg':
            table = self._reg_table.copy()
        else:
            raise ValueError(
                '{} should be one of group or corr'.format(table_type))
        
        if not bg:
            bg = table.index.tolist()
        elif bg.lower() == 'no':
            bg = None

        if isinstance(query_condition, str):
            query_condition = {'query': query_condition}
        o = ora(tool)
        self._enrich_tool['ora'] = tool

        out_combine = pd.DataFrame()
        for annot, qc in query_condition.items():
            enrichment_genes = table.query(
                qc).index.tolist()
            out = o.run(enrichment_genes, organism=organism,
                       bg=bg, db=db, **kwargs)
            out.insert(0, 'annotation', annot)
            out_combine = pd.concat([out_combine, out])
        self.ora = out_combine

    def gsea(self,
             table_type,
             db=None,
             value_label=None,
             query_condition=None,
             permutation_type='phenotype',
             threads=6,
             **kwargs):
        self._enrich_tool['gsea_'] = 'gsea_'
        if not db:
            raise ValueError('Please set the db parameter as any of "{}"'.format(
                ', '.join(gp.get_library_name())))
        if isinstance(db, str):
            db = [db]

        if table_type == 'group':
            data = self._tmp_dataset.copy()
            cls = self._data[self._file_type].loc[self._tmp_dataset.columns, self._group_name].values.tolist()
            out_combine = pd.DataFrame()
            for database in db:
                gs_res = gp.gsea(
                    data=data,
                    gene_sets=database,
                    cls=cls,
                    permutation_type=permutation_type,
                    threads=threads,
                    **kwargs
                )
                out_combine = pd.concat([out_combine, gs_res.res2d])
            out_combine.loc[out_combine[out_combine['NES']>0].index, 'annotation'] = cls[0]
            out_combine['annotation'] = out_combine['annotation'].fillna(cls[-1])
            columns = ['annotation']
            columns.extend(out_combine.columns[:-1])
            self.gsea_ = out_combine.copy()[columns].sort_values('NES', ascending=False)
        else:

            if table_type == 'corr':
                predata = self._corr_table.copy()
            elif table_type == 'reg':
                predata = self._reg_table.copy()

            rnk = predata[[value_label]].sort_values(by=value_label, ascending=False)
            if query_condition:
                rnk = rnk.query(query_condition)

            self.gsea_ = pd.DataFrame()
            for database in db:
                pre_res = gp.prerank(
                    rnk=rnk,
                    gene_sets=database,
                    threads=threads,
                    **kwargs)
                self.gsea_ = pd.concat([self.gsea_, pre_res.res2d])

    def count(self,
              impute_value='min',
              axis='columns',
              figsize=(3, 2.5),
              addline='median',
              fmt='.0f',
              title=None,
              labels_hide=['x', 'y'],
              return_data=True,
              save=None,
              outname=None,
              **kwargs):
        if impute_value == 'min':
            impute_value = self._data[self._dataset_type].min()

        axis = {'columns': 0, 'rows': 1}.get(axis)

        addline_method = {'mean': np.mean,
                          'median': np.median}.get(addline, False)

        plotdata = self._tmp_dataset.replace(
            impute_value, np.nan).count(axis=axis).rename('count')

        if self._group_name:
            tmp_value = self._data[self._file_type][self._group_name]
            plotdata = pd.concat([plotdata, tmp_value], axis=1, join='inner').set_index(
                self._group_name, append=True).reset_index()
        index_ = int(np.where(plotdata.columns==self._group_name[0])[0][0])
        plotdata.insert(index_, 'hue', plotdata[self._group_name].apply(lambda x: '_'.join(x), axis=1))
        plotdata = plotdata.drop(self._group_name, axis=1)

        ax = barplot(plotdata,
                     palette=self._palette,
                     title=title,
                     labels_hide=labels_hide,
                     figsize=figsize,
                     **kwargs)
        ax.set_xticks([])
        if addline:
            if isinstance(plotdata, pd.DataFrame):
                value = addline_method(plotdata['count'])
            else:
                value = addline_method(plotdata)
            ax.axhline(y=value, c="black", ls="--", dashes=(11, 8), lw=0.5)
            ax.text(x=len(plotdata), y=value, s=': '.join(
                (addline, ("{:" + fmt + "}").format(value))), horizontalalignment='right', verticalalignment='bottom')
        if save:
            if title and not outname:
                outname = title

            savefig(ax, outpath=os.path.join(self._dirpath, 'figure'), outname=outname, plot_type=sys._getframe(
            ).f_code.co_name, out_format=save)
        if return_data:
            return plotdata, ax
        else:
            return ax

    def accumulative(self,
                     impute_value='min',
                     axis='columns',
                     figsize=(3, 2.5),
                     title=None,
                     labels_hide=['x', 'y'],
                     save=None,
                     outname=None,
                     **kwargs):
        if impute_value == 'min':
            impute_value = self._data[self._dataset_type].min()
        else:
            pass

        pre_plotdata = self._tmp_dataset.replace(
            impute_value, np.nan).applymap(lambda x: 1 if x == x else 0)

        # if self._part_element and self._palette:
        #     self._palette = dict(
        #         zip(self._part_element, [self._palette[i] for i in self._part_element]))
        # else:
        #     pass

        plotdata = pre_plotdata.cumsum(axis=axis).applymap(
            lambda x: 1 if x != 0 else 0).sum()

        if self._group_name:
            tmp_value = self._data[self._file_type][self._group_name]

            plotdata = pd.concat([plotdata, tmp_value], axis=1, join='inner').set_index(
                self._group_name, append=True).reset_index()
        else:
            pass

        ax = scatterplot(plotdata,
                         palette=self._palette,
                         title=title,
                         labels_hide=labels_hide,
                         figsize=figsize,
                         **kwargs)
        ax.set_xticks([])
        if save:
            if title and not outname:
                outname = title
            else:
                pass
            savefig(ax, outpath=os.path.join(self._dirpath, 'figure'), outname=outname, plot_type=sys._getframe(
            ).f_code.co_name, out_format=save)

        return ax

    def range(self,
              axis='columns',
              method='mean',
              c=['grey'],
              s=5,
              highlight_annots={},
              figsize=(4, 2.5),
              ylabel='$\mathregular{Log_{10}}$(FoT)',
              title=None,
              labels_hide=['x'],
              save=None,
              outname=None,
              **kwargs):
        if method not in ['mean', 'median']:
            raise ValueError(
                'rangeplot function only support mean and median parameter now.')
        else:
            pass

        if self._group_name:
            min_value = np.log10(self._data[self._dataset_type].min().min())
            plotdata = pd.DataFrame(np.log10(self._Group__param_values[method].T), columns=self._group_values, index=self._tmp_dataset.index).stack().astype('float32').rename_axis(['Symbol', self._group_name]).groupby(
                self._group_name).apply(lambda x: x.sort_values(ascending=False).rename(ylabel).reset_index()).droplevel(0).set_index('Symbol', append=True).droplevel(0).replace(min_value, np.nan).dropna()
            plotdata = plotdata.groupby('cohort').apply(lambda x: x.reset_index().rename_axis('rank').reset_index().set_index(plotdata.index.name))
            plotdata['rank'] = plotdata['rank'] + 1
        else:
            stand_method = {'mean': np.mean,
                            'median': np.median}.get(method, None)
            plotdata = np.log10(stand_method(self._data[self._dataset_type], axis=axis)).sort_values(
                ascending=False).rename(ylabel).to_frame()
            plotdata.insert(0, 'rank', range(1, len(plotdata)+1))

        palette = self._palette.copy()

        if highlight_annots:
            for k, v in highlight_annots.items():
                highlight_points, color, size = v
                plotdata.loc[highlight_points, ('hue', 'size')] = (k, size)
                palette[k] = color
            plotdata.loc[:, 'hue'] = plotdata.loc[:, 'hue'].fillna('others')
            plotdata.loc[:, 'size'] = plotdata.loc[:, 'size'].fillna(s)
            palette['others'] = c[0]
            plotdata = plotdata.iloc[:, [0, 2, 3, 4]]
        else:
            highlight_points = []

        

        ax = scatterplot(plotdata,
                         figsize=figsize,
                         palette=palette,
                         title=title,
                         labels_hide=labels_hide,
                         highlight_points=highlight_points,
                         **kwargs)
        if save:
            if title and not outname:
                outname = title
            else:
                pass
            savefig(ax, outpath=os.path.join(self._dirpath, 'figure'), outname=outname, plot_type=sys._getframe(
            ).f_code.co_name, out_format=save)

        return ax

    def overlap(self, title=None, save=None, outname=None, **kwargs):
        if not self._group_name:
            raise ValueError(
                'Please set group_name parameter by set_param function.')
        else:
            plotdata = [np.take(self._tmp_dataset.index, np.nonzero(i))[
                0] for i in self._Group__param_values['percentage']]

            ax = vennplot(plotdata, labels_name=self._group_values,
                          palette=[self._palette[i] for i in self._group_values], title=title, **kwargs)

            if save:
                if title and not outname:
                    outname = title
                else:
                    pass
                savefig(ax, outpath=os.path.join(self._dirpath, 'figure'), outname=outname, plot_type=sys._getframe(
                ).f_code.co_name, out_format=save)

            return ax

    def cate(self, elements, data_type=None, data_type_annot='auto', method='ranksums', value_log_transform=None, quantile=False, ax=None, figsize=(1.8, 1.8), one_plot=False, category_type=['violin', 'strip'], orient='v', title=None, ticklabels_format=['y'], ticklabels_wrap=[], wrap_length=None, one_pdf=False, save=False, outname=None, return_data=False, **kwargs):
        if isinstance(elements, str):
            elements = [elements]
        if not data_type:
            data_type = [self._dataset_type]
        data = self.merge_data_group(elements, data_type=data_type).dropna(how='all', axis=1)
        if len(self._group_name) > 1:
            data.index = pd.MultiIndex.from_tuples(zip(data.index.get_level_values(0), data.index.map(lambda x: '_'.join(x[1:]))), names=(data.index.names[0], '_'.join(data.index.names[1:])))
        
        if (data_type_annot == 'no') or (data_type_annot == 'auto' and len(data_type) == 1):
            data = data.rename(columns=lambda x: x.split('|')[0])
            elements_loop = elements
        else:
            elements_loop = list(['|'.join(i) for i in product(elements, data_type)])
        
        if len(elements_loop) == 1:
            order = self._part_element
            name = elements[0]
        elif not one_plot:
            order = self._part_element
            name = ''
        else:
            order = elements_loop
            name = ' '
        
        data = data.stack().swaplevel(1, 2).rename(name).astype(float)
        if quantile:
            if isinstance(self._group_name, str):
                group_name = [self._group_name]
            else:
                group_name = self._group_name
            groupby_element = '_'.join(group_name)
            data = data.groupby(groupby_element).apply(lambda x: filter_by_quantile(x)).droplevel(0)
        
        if not title:
            title = name
        
        _, _, hue = data.index.names

        if value_log_transform:
            assert value_log_transform in ['log2', 'log10'], "parameter value_log_transform should be one of 'log2' and 'log10'"
            data = {'log2': np.log2, 'log10': np.log10}.get(value_log_transform)(data)
            kwargs['log_transform'] = 'no'

        params = {'hue_order': self._part_element, 'method': method, 'category_type': category_type, 'palette': self._palette, 'ax': ax, 'figsize': figsize, 'title': title, 'orient': orient, 'ticklabels_format': ticklabels_format, 'ticklabels_wrap': ticklabels_wrap, 'wrap_length': wrap_length}
        
        if one_plot:
            axs = cateplot(data, hue=hue, order=order, **params, **kwargs)
        else:
            axs = []
            for element in elements_loop:    
                tmp_data = data.xs(element, axis=0, level=1, drop_level=False).rename(element)
                params['title'] = element
                axs.append(cateplot(tmp_data, order=order,
                          hue=hue, **params, **kwargs))

        if save:
            if title and not outname:
                outname = title
            else:
                pass
            savefig(ax, outpath=os.path.join(self._dirpath, 'figure'), outname=outname, plot_type=sys._getframe(
            ).f_code.co_name, out_format=save)
        if return_data:
            return data, axs
        else:
            return axs

    def heat(self, elements, annot_dict=None, lut=None, group_name=None, data_type=None, data_type_annot='auto', join_method='inner', sort='element', sort_group=None, z_score=0, shuffle=False, return_data=False, **kwargs):
        if not data_type:
            data_type = [self._dataset_type]
        if not group_name:
            if isinstance(self._group_name, str):
                group_name = [self._group_name]
            else:
                group_name = self._group_name
        
        plotdata = self.merge_data_group(elements, group_name=group_name, data_type=data_type, join_method=join_method, sort=sort, sort_group=sort_group, shuffle=shuffle).astype(float)

        if annot_dict:
            tmp_columns = plotdata.columns.str.split('|', expand=True).get_level_values(0)
            plotdata.columns = pd.MultiIndex.from_arrays(np.vstack((tmp_columns, [tmp_columns.map(v) for k, v in annot_dict.items()])), names=np.hstack(('Genes', list(annot_dict.keys()))))

        remove = False
        if (data_type_annot == 'no') or (data_type_annot == 'auto' and len(data_type) == 1):
            remove = True
        if remove:
            plotdata = plotdata.rename(columns=lambda x: x.split('|')[0], level=0)

        if len(group_name) > 1:
            plotdata_index_element = dict(zip(plotdata.index.names[1:], [i.values for i in plotdata.droplevel(0).index.levels]))
            lut_default = {i: {k: v for k, v in j.items() if k in plotdata_index_element[i]} for i, j in self._color_map.items() if i in group_name}
        else:
            lut_default = {i: {k: v for k, v in j.items() if k in self._part_element} for i, j in self._color_map.items() if i in group_name}

        if lut:
            lut_default.update(lut)
        
        if 'col_cluster' in kwargs.keys() or 'row_cluster' in kwargs.keys():
            plotdata = plotdata.fillna(1e-5)
        
        if any([kwargs.get('col_cluster', None), kwargs.get('row_cluster', None)]):
                plotdata = plotdata.fillna(1e-5)

        ax = heatmap(plotdata.T, lut=lut_default, z_score=z_score, **kwargs)
        if return_data:
            return plotdata, ax
        return ax

    def scatter(self, elements, volcano=False, sig_log_transform=True, hue=None, size=None, highlight_points=None, palette=None, ax=None, figsize=(1.8, 1.8), title=None, adjust_axes=True, ticklabels_hide=[], ticklabels_format=['y'], ticklabels_wrap=[], wrap_length=None, spines_hide=[], labels_hide=[], return_data=True, **kwargs):
        series_list = []
        for table_name in ['_group_table', '_corr_table', '_reg_table']:
            if hasattr(self, table_name):
                add_new = [self.__dict__[table_name][element] for element in elements if element in self.__dict__[table_name].columns]
                series_list.extend(add_new)
        table = pd.concat(series_list, axis=1, join='inner').reindex(elements, axis=1).dropna(how='all', axis=1)
        if len(elements) == 3:
            table = table.iloc[:, [0, 2, 1]]

        columns_shape = table.columns.shape[0]
        if table.columns.nunique() != columns_shape:
            if columns_shape > 2:
                rename_column = table.columns[2]
                table = pd.concat([table.iloc[:, :2], table.iloc[:, 2:].rename(columns={rename_column: rename_column+'_hue'})], axis=1)
            if columns_shape > 3:
                rename_column = table.columns[3]
                table = pd.concat([table.iloc[:, :3], table.iloc[:, 3:].rename(columns={rename_column: rename_column+'_size'})], axis=1)

        if volcano:
            sig_up_color, sig_down_color = self._palette[self._dividend], self._palette[self._divisor]
            if not title:
                title = '{} vs. {}'.format(self._dividend, self._divisor)
            if isinstance(highlight_points, dict):
                highlight_points = np.unique(np.hstack(self.get_gene_from_enrichment(highlight_points).values())).tolist()
            
            out_ = volcanoplot(table, title=title, sig_up_color=sig_up_color, sig_down_color=sig_down_color, highlight_points=highlight_points, adjust_axes=adjust_axes, ticklabels_hide=ticklabels_hide, ticklabels_format=ticklabels_format, ticklabels_wrap=ticklabels_wrap, wrap_length=wrap_length, spines_hide=spines_hide, labels_hide=labels_hide, ax=ax, figsize=figsize, return_data=return_data, **kwargs)
            if return_data:
                table, ax = out_
            else:
                ax = out_
        else:
            if sig_log_transform:
                if not isinstance(sig_log_transform, Iterable):
                    sig_log_transform = table.filter(regex="pvalues|fdr").columns
                table[sig_log_transform] = -np.log10(table[sig_log_transform])
                def rename_columns(x):
                    if x in sig_log_transform:
                        x = '-Log10({})'.format(x)
                    return x.replace('_', ' ').capitalize()

                table = table.rename(columns=lambda x: rename_columns(x))

            if hue:
                if isinstance(hue, (pd.Series)):
                    hue = hue.to_frame()

                if isinstance(hue, (pd.DataFrame)):
                    hue = hue.to_dict()

                if isinstance(hue, dict):
                    if not any(isinstance(i, dict) for i in hue.values()):
                        hue = {'hue': hue}
                    k, v = list(hue.items())[0]
                    v = self.get_gene_from_enrichment(v)

                for k1, v1 in v.items():
                    gene = np.intersect1d(table.index, v1)
                    table.loc[gene, k] = k1
                table[k] = table[k].fillna('')
                if size:
                    if size == 'hue':
                        table.loc[:, 'size'] = table[k].apply(lambda x: 'annot' if x!='' else x)
                    else:
                        if any(isinstance(i, dict) for i in size.values()):
                            k, v = list(size.items())[0]
                        else:
                            k = 'size'
                            v = size
                        table.loc[:, k] = table.index.map(v).fillna('')
                else:
                    table = table.iloc[:, [0, 2, 1]]
                table = table.sort_values(k, ascending=True)
            
            if highlight_points == 'hue':
                highlight_points = np.hstack(list(v.values())).tolist()
            
            ax = scatterplot(table, title=title, palette=palette, highlight_points=highlight_points, adjust_axes=adjust_axes, ticklabels_hide=ticklabels_hide, ticklabels_format=ticklabels_format, ticklabels_wrap=ticklabels_wrap, wrap_length=wrap_length, spines_hide=spines_hide, labels_hide=labels_hide, ax=ax, figsize=figsize, **kwargs)
        if return_data:
            return table, ax
        else:
            return ax

    def bubble(self, elements, query_condition=None, sort=None, ascending=True, sig_log_transform=True, highlight_points=None, palette=R_CMAP, ax=None, figsize=(1.8, 1.8), title=None, adjust_axes=True, ticklabels_hide=[], ticklabels_format=[], ticklabels_wrap=[], wrap_length=None, spines_hide=[], labels_hide=[], **kwargs):
        table_name, elements = list(elements.items())[0]
        series_list = [self.__dict__[table_name][element] for element in elements if element in self.__dict__[table_name].columns]
        table = pd.concat(series_list, axis=1, join='inner')[elements]
        if len(elements) == 3:
            table = table.iloc[:, [0, 2, 1]]

        if query_condition:
            table = table.query(query_condition)
        if sort:
            table = table.groupby(elements[0], as_index=False).apply(lambda x: x.sort_values(by=sort, ascending=ascending)).reset_index(drop=True)
            
        if sig_log_transform:
            if not isinstance(sig_log_transform, Iterable):
                sig_log_transform = table.filter(regex="pvalue|fdr|FDR|qvalue|adjust|p-val|p_value").columns
            table[sig_log_transform] = -np.log10(table[sig_log_transform].astype(float))
            def rename_columns(x):
                if x in sig_log_transform:
                    x = '-Log10({})'.format(x)
                return x.replace('_', ' ').capitalize()

            table = table.rename(columns=lambda x: rename_columns(x))
            
        ax = scatterplot(table, title=title, palette=palette, highlight_points=highlight_points, adjust_axes=adjust_axes, ticklabels_hide=ticklabels_hide, ticklabels_format=ticklabels_format, ticklabels_wrap=ticklabels_wrap, wrap_length=wrap_length, spines_hide=spines_hide, labels_hide=labels_hide, ax=ax, figsize=figsize, **kwargs)
        ax.set_xlim(ax.get_xlim()[0]-.3, ax.get_xlim()[1]+.3)
        return ax

    def bar(self):
        pass
