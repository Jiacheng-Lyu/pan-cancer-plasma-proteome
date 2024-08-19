import os
from itertools import product
from collections import defaultdict

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.colors as mcolors

from gprofiler import GProfiler
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri

numpy2ri.activate()
pandas2ri.activate()

robjects.r.source(os.path.join(os.path.dirname(__file__), 'R', 'R_tools.R'))

def rec_dd():
    return defaultdict(rec_dd)


def r_func(func_name, *args, **kwargs):
    r_func_dict = {
        'clusterkegg': robjects.r.clusterkegg,
        'clustermsigdb': robjects.r.clustermsigdb,
        'clustergo': robjects.r.clustergo,
        'clusterreactome': robjects.r.clusterreactome,
        'ssgsea': robjects.r.ssgsea,
        'sumer_usage': robjects.r.sumer_usage,
        'combat_batch': robjects.r.combat_batch,
        'dip': robjects.r.dip_test,
        'impute': robjects.r.impute, 
        'ccp': robjects.r.ccp
    }
    if func_name in [
            'clusterkegg', 'clustermsigdb', 'clustergo', 'clusterreactome'
    ]:
        object_ = robjects.FactorVector(*args)
        if 'universe' in kwargs:
            kwargs['universe'] = robjects.FactorVector(kwargs['universe'])

    elif func_name in ['ssgsea', 'impute', 'ccp']:
        object_ = robjects.DataFrame(*args)

    elif func_name in ['sumer_usage']:
        object_ = robjects.StrVector(args)

    elif func_name in ['combat_batch']:
        object_ = robjects.DataFrame(*args)
        kwargs['batch'] = robjects.FactorVector(kwargs['batch'])
        if 'mod' in kwargs:
            kwargs['mod'] = robjects.FactorVector(kwargs['mod'])

    elif func_name in ['dip']:
        object_ = robjects.FloatVector(*args)
    r_re = r_func_dict[func_name](object_, **kwargs)
    if type(r_re) == np.recarray:
        r_re = pd.DataFrame(r_re)
    re = robjects.conversion.rpy2py(r_re)

    return re


def filter_by_quantile(x, up=0.75, bottom=0.25):
    '''
    Filter data by its quantile(IQR);
    :param x: pd.Series 
    :param up: upper quantile
    :param bottom: lower quantile

    return: a filtered series;
    '''

    q1 = x.quantile(bottom)
    q2 = x.quantile(up)
    iqr = q2 - q1
    filter = (x >= q1 - 1.5 * iqr) & (x <= q2 + 1.5 * iqr)
    return x.loc[filter]


def percentage(x, axis=0, min=None):
    x = x.copy()
    if isinstance(min, (int, float)):
        x[x == min] = np.nan
    elif np.isnan(x).sum() > 0:
        pass
    else:
        return np.repeat(np.array([1]), x.shape[0])

    if x.ndim == 1:
        axis += 1
        x = x[None, :]
    return np.sum(~np.isnan(x), axis=axis)/x.shape[1]


def dropnan(df, min_num=None, thresh=.5, axis=0):
    if min_num:
        df = df.replace(min_num, np.nan)
    df = df.dropna(thresh=math.ceil(thresh * df.shape[1]), axis=axis)
    if min_num:
        df = df.fillna(min_num)
    return df


def dateset_preprocess_for_statistic(df, prestatistic_method=None):
    prestatistic_methods = {'log2': np.log2, 'log10': np.log10}
    if isinstance(df, (pd.Series, pd.DataFrame)):
        group_values = [
            group.values for _, group in df.groupby(df.index)[df.columns[0]]
        ]
        if group_values[0].ndim == 1:
            group_values = list(map(lambda x: x[None, :], group_values))
    if prestatistic_method in prestatistic_methods.keys():
        statistic_data = list(
            map(lambda x: prestatistic_methods[prestatistic_method](x),
                group_values))
    else:
        statistic_data = group_values

    return statistic_data


def handle_colors(df, continuous_value=None):
    color_dict = defaultdict(dict)
    df1 = df[~df.index.isin(['continuous'], level=1)]
    color_dict.update(df1.groupby(df1.index.names[0]).apply(
        lambda x: x.loc[x.name].to_dict()[df1.columns[0]]).to_dict())
    df2 = df[df.index.isin(['continuous'], level=1)]
    if not all(df2.shape):
        pass
    else:
        for i, label in enumerate(df2.index):
            colormap = mcolors.LinearSegmentedColormap.from_list("red_blue_gradient", list(
                map(lambda x: x.strip(), df2.iloc[i, 0].split(','))), N=256)
            if df2.index[i][0] not in continuous_value.columns:
                raise ValueError(
                    "{} not in continuous_value DataFrame, Please check ").format()
            value = continuous_value[df2.index[i][0]].dropna().values
            scaler = list(map(lambda x: colormap(x), MinMaxScaler(
            ).fit_transform(value[None, :].T).T[0].tolist()))
            color_dict[label[0]] = dict(zip(value, scaler))
    return color_dict


def sort_custom(item, label, order):
    reorder_columns = True
    if isinstance(item, pd.Series):
        df = item.to_frame()
        if not df.index.name:
            df.index.name = 'ID'

        reorder_columns = False
    else:
        df = item.copy()

    if isinstance(label, str):
        label = [label]

    set_index = False
    if not isinstance(item.index, pd.RangeIndex):
        df = df.reset_index()
        set_index = True

    if len(label) != 1:
        new_index = list(product(*order))
    else:
        new_index = order
    
    df = df.set_index(label)
    new_index = [i for i in new_index if i in df.index.unique()]
    df = df.loc[new_index].reset_index()
    
    if set_index:
        df = df.set_index(item.index.names)

    if reorder_columns:
        df = df[item.columns]
    else:
        df = df.loc[:, label[0]]

    return df


def large_than_n_value(df, n=2, ratio=1):
    if n > df.shape[1]:
        raise ValueError('The number of n should less than the {}'.format(df.shape[1]))
    n_largest_ = np.sort(df, axis=1)[:, -n] * ratio
    n_largest_spread = np.repeat(n_largest_[:, None], repeats=df.shape[1], axis=1)
    df_compare = np.greater(df, n_largest_spread)
    return df_compare


class ora:
    def __init__(self, tool):
        tools = ['gprofiler', 'clusterprofiler']
        if tool not in tools:
            raise ValueError(
                'ORA tool parameter should be one of {}'.format(', '.join(tools)))
        else:
            self.tool = tool
            self.db = []

    def run(self, gene, organism='hsapiens', bg=None, db=None, **kwargs):
        self.db = []
        if isinstance(db, str):
            db = [db]
        if self.tool == 'gprofiler':
            databases = {
                'gobp': 'GO:BP',
                'gomf': 'GO:MF',
                'gocc': 'GO:CC',
                'kegg': 'KEGG',
                'reac': 'REAC'}
            self._check_parameter(db, databases)
            gp = GProfiler(return_dataframe=True)
            out = gp.profile(organism=organism, query=gene, background=bg,
                             sources=self.db, no_evidences=False, **kwargs)
            return out
        elif self.tool == 'clusterprofiler':
            databases = {
                'kegg': 'clusterkegg',
                'go': 'clustergo',
                'reac': 'clusterreactome',
                'hallmark': 'clustermsigdb'
            }
            if not db:
                self.db = list(databases.values())
            else:
                self._check_parameter(db, databases)

            out = pd.DataFrame()
            if not bg:
                bg = robjects.NULL
            for database in self.db:
                en_res = r_func(database,
                                gene, universe=bg, **kwargs)
                out = pd.concat([out, en_res])
            return out

    def _check_parameter(self, check_values, background_values):
        if check_values:
            for check_value in check_values:
                if check_value.lower() not in background_values:
                    raise ValueError('{} not in {}, please check your parameter'.format(
                        check_value, background_values))
                else:
                    self.db.append(background_values[check_value])