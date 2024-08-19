import itertools
from collections import Iterable

import numpy as np
import pandas as pd

import scipy.stats

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn3, venn2_unweighted, venn3_unweighted

from .base import (check_type, _plotdata_handle, _heatmap_legend_handle, axes_, add_stats, adjusttext, LegendTitle)
from .constants import RB_CMAP, MCMAP
from utils.function import dateset_preprocess_for_statistic, sort_custom
from utils.statistic import statistic_func


@check_type
def barplot(df,
            palette=None,
            title=None,
            orient='v',
            ax=None,
            figsize=(3, 3),
            adjust_axes=True,
            ticklabels_hide=['x'],
            ticklabels_format=['y'],
            ticklabels_wrap=['y'],
            spines_hide=['top', 'right'],
            labels_hide=None,
            linewidth=0,
            wrap_length=20,
            dodge=False,
            **kwargs):
    """
    Create a bar plot of the given data.

    Parameters
    ----------
    df : pd.DataFrame
        The data to plot.
    palette : str, optional
        The color palette to use for the plot. Default is None.
    title : str, optional
        The title of the plot. Default is None.
    orient : str, optional
        The orientation of the plot. Can be either 'v' for vertical or 'h' for horizontal. Default is 'v'.
    ax : plt.Axes, optional
        The Axes object to draw the plot on. If not provided, a new one will be created.
    figsize : Tuple[int, int], optional
        The size of the figure to create if no Axes object is provided. Default is (3, 3).
    adjust_axes : bool, optional
        Whether to adjust the formatting of the axes. Default is True.
    ticklabels_hide : List[str], optional
        The tick labels to hide. Default is ['x'].
    ticklabels_format : List[str], optional
        The format of the tick labels. Default is ['y'].
    spines_hide : List[str], optional
        The spines to hide. Default is ['top', 'right'].
    labels_hide : List[str], optional
        The axis labels to hide. Default is None.
    linewidth : float, optional
        The width of the lines around the bars. Default is 0.
    dodge : bool, optional 
        When hue nesting is used, whether elements should be shifted along the categorical axis.
    **kwargs
        Additional keyword arguments passed to sns.barplot.

    Returns
    -------
    plt.Axes
        The Axes object on which the plot was drawn.
    """

    
    df, label, x, y, hue, args = _plotdata_handle(df, palette)
    if orient == 'h':
        x, y = y, x
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    ax = sns.barplot(data=df,
                     x=x,
                     y=y,
                     hue=hue,
                     palette=palette,
                     ax=ax,
                     linewidth=linewidth,
                     dodge=dodge,
                     **kwargs)
    if adjust_axes:
        ax = axes_(ax,
                   title,
                   ticklabels_hide=ticklabels_hide,
                   ticklabels_format=ticklabels_format,
                   spines_hide=spines_hide,
                   labels_hide=labels_hide,
                   ticklabels_wrap=ticklabels_wrap, 
                   wrap_length=wrap_length)
    return ax


@check_type
def scatterplot(df,
                title=None,
                palette=None,
                ax=None,
                figsize=(1.8, 1.8),
                linewidth=0,
                hue_order=None,
                size=None,
                sizes=None,
                style=None,
                highlight_points=None,
                adjust_axes=True,
                ticklabels_hide=['x'],
                ticklabels_format=['y'],
                ticklabels_wrap=['y'],
                wrap_length=20,
                spines_hide=['top', 'right'],
                labels_hide=None,
                legend='brief',
                text_label=None,
                **kwargs):
    """
    Create a scatter plot of the given data.

    Parameters
    ----------
    df : pd.DataFrame
        The data to plot.
    title : str, optional
        The title of the plot. Default is None.
    palette : str, optional
        The color palette to use for the plot. Default is None.
    ax : plt.Axes, optional
        The Axes object to draw the plot on. If not provided, a new one will be created.
    figsize : Tuple[int, int], optional
        The size of the figure to create if no Axes object is provided. Default is (3, 3).
    linewidth : float, optional
        The width of the lines around the scatter points. Default is 0.
    hue_order : List[str], optional
        The order in which to plot the hue levels. Default is None.
    size : str, optional
        The column in the DataFrame to use for sizing the scatter points. Default is None.
    sizes : Tuple[float, float], optional
        The minimum and maximum size of the scatter points. Default is None.
    style : str, optional
        The column in the DataFrame to use for styling the scatter points. Default is None.
    highlight_points : List[str], optional
        A list of points to highlight on the plot. Default is None.
    adjust_axes : bool, optional
        Whether to adjust the formatting of the axes. Default is True.
    ticklabels_hide : List[str], optional
        The tick labels to hide. Default is ['x'].
    ticklabels_format : List[str], optional
        The format of the tick labels. Default is ['y'].
    ticklabels_wrap : List[str], optional
        The tick labels to wrap. Default is ['y'].
    wrap_length : int, optional
        The maximum length of wrapped tick labels. Default is 20.
    spines_hide : List[str], optional
        The spines to hide. Default is ['top', 'right'].
    labels_hide : List[str], optional
        The axis labels to hide. Default is None.
    legend : str, optional
        How to draw the legend. Default is 'brief'.
    text_label : str, optional
        The column in the DataFrame to use for labeling points on the plot. Default is None.
    **kwargs
        Additional keyword arguments passed to sns.scatterplot.

    Returns
    -------
    plt.Axes
        The Axes object on which the plot was drawn.
    """


    df, label, x, y, hue, size = _plotdata_handle(df)

    if sizes and not size:
        size = hue
    elif not sizes and size:
        sizes = (5, 15)

    if not ax:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if not text_label:
        text_label = df.columns[0]

    ax = sns.scatterplot(data=df,
            x=x,
            y=y,
            hue=hue,
            palette=palette,
            ax=ax,
            hue_order=hue_order,
            size=size,
            sizes=sizes,
            style=style,
            linewidth=linewidth,
            legend=legend,
            **kwargs)

    if highlight_points:
        if df.shape[1] == 2:
            annot_df = df.reset_index().set_index(x)
            annot_df.columns = x, y
        else:
            annot_df = df.set_index(text_label)
        adjusttext(annot_df, ax, highlight_points, x, y)

    if adjust_axes:
        ax = axes_(ax,
                   title,
                   ticklabels_hide=ticklabels_hide,
                   ticklabels_format=ticklabels_format,
                   ticklabels_wrap=ticklabels_wrap,
                   wrap_length=wrap_length,
                   spines_hide=spines_hide,
                   labels_hide=labels_hide)
    return ax


@check_type
def lineplot(df,
            title=None,
            palette=None,
            ax=None,
            figsize=(3, 3),
            hue_order=None,
            style=None,
            adjust_axes=True,
            ticklabels_hide=['x'],
            ticklabels_format=['y'],
            ticklabels_wrap=['y'],
            wrap_length=20,
            spines_hide=['top', 'right'],
            labels_hide=None,
            legend='brief',
            text_label=None,
            **kwargs):
    df, label, x, y, hue, size = _plotdata_handle(df)

    if size:
        sizes = (df[size].min(), df[size].max())
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=figsize)


    if not text_label:
        text_label = df.columns[0]
    ax = sns.lineplot(data=df,
            x=x,
            y=y,
            hue=hue,
            palette=palette,
            ax=ax,
            hue_order=hue_order,
            size=size,
            style=style,
            legend=legend,
            **kwargs)

    if adjust_axes:
        ax = axes_(ax,
                   title,
                   ticklabels_hide=ticklabels_hide,
                   ticklabels_format=ticklabels_format,
                   ticklabels_wrap=ticklabels_wrap,
                   wrap_length=wrap_length,
                   spines_hide=spines_hide,
                   labels_hide=labels_hide)
    return ax


@check_type
def cateplot(df,
             x=None,
             hue=None,
             order=None,
             hue_order=None,
             title=None,
             palette=None,
             ax=None,
             figsize=(2.5, 3),
             width=0.8,
             category_type=['box', 'strip'],
             inner=None,   # violin_linewidth should > 0
             violin_linewidth=0,
             violinalpha=0.3,
             scale='width', 
             showfliers=False,
             showcaps=False,
             box_pairs='All',
             probs=None,
             box_facecolor='none',
             stripsize=6,
             stripalpha=0.8,
             orient='v',
             log_transform='log2',
             method='ttest',
             one_tail=False,
             text_format='star',
             adjust_axes=True,
             ticklabels_hide=None,
             ticklabels_format=['y'],
             ticklabels_wrap = [],
             wrap_length=None,
             spines_hide=['top', 'right', 'bottom'],
             labels_hide=['y'],
             ttest_kwargs={},
             **kwargs):

    df, label, x, y, hue, size = _plotdata_handle(df)

    if df[x].nunique() == 1:
        x = hue
        hue = None

    if order:
        df = sort_custom(df, order=order, label=x)
    
    order = pd.unique(df[x])
    
    if orient == 'h':
        x, y = y, x

    if not ax:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    
    if 'violin' in category_type:
        plot = 'violinplot'
        ax = sns.violinplot(data=df,
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            width=width,
            palette=palette,
            linewidth=violin_linewidth,
            inner=inner,
            scale=scale,
            orient=orient,
            ax=ax,
            dodge=True,
            **kwargs
        )
        plt.setp(ax.collections, alpha=violinalpha)

    if 'box' in category_type:
        plot = 'boxplot'
        if not hue or 'strip' in category_type:
            box_props = {
                'boxprops': {
                    'facecolor': box_facecolor,
                    'edgecolor': 'k'
                },
                'medianprops': {
                    'color': 'k'
                },
                'whiskerprops': {
                    'color': 'k'
                },
                'capprops': {
                    'color': 'k'
                }
            }
        else:
                box_props = {}
            
        ax = sns.boxplot(data=df,
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            width=width,
            palette=palette,
            showfliers=showfliers,
            showcaps=showcaps,
            orient=orient,
            ax=ax,
            dodge=True,
            **box_props,
            **kwargs
        )

    if 'strip' in category_type:
        plot = 'stripplot'
        # strip_kwargs = {}
        # if hue:
        #     strip_kwargs['dodge'] = True

        ax = sns.stripplot(data=df,
            x=x,
            y=y,
            order=order,
            hue=hue,
            hue_order=hue_order,
            palette=palette,
            alpha=stripalpha,
            size=stripsize,
            orient=orient,
            ax=ax,
            dodge=True
        )

    if box_pairs:
        if not probs:
            if isinstance(box_pairs, str) and box_pairs.lower() == 'all':
                if not hue:
                    box_pairs = list(itertools.combinations(np.unique(df[x]), 2))
                else:
                    list1 = df[x].unique().tolist()
                    list2 = df[hue].unique().tolist()
                    box_pairs = [((i, j[0]), (i, j[1])) for i in list1 for j in list(itertools.combinations(list2, 2))]
                    print(list1, list2)
            if not hue:
                set_index = x
            else:
                set_index = [x, hue]
            
            probs = [statistic_func(df.set_index(set_index).loc[list(i), [y]].pipe(dateset_preprocess_for_statistic, prestatistic_method=log_transform), statistic_method=method, ttest_kwargs=ttest_kwargs)[1][0] for i in box_pairs]

            if one_tail:
                probs = [prob/2 for prob in probs]
        
        add_stats(ax, df, x, y,
            hue=hue,
            plot=plot,
            box_pairs=box_pairs,
            line_offset=0.1,
            text_offset=3,
            probs=probs, 
            orient=orient,
            test_short_name=method,
            text_format=text_format,
            legend=False
        )

    if not title:
        title = y

    if adjust_axes:
        ax = axes_(ax, title,
            ticklabels_hide=ticklabels_hide,
            ticklabels_format=ticklabels_format,
            spines_hide=spines_hide,
            labels_hide=labels_hide,
            ticklabels_wrap=ticklabels_wrap,
            wrap_length=wrap_length,
            legend=False
        )

    return ax


@check_type
def regplot(df,
            method='spearman',
            scattersize=20,
            scattercolor='black',
            linecolor='red',
            ax=None,
            figsize=(3, 3),
            adjust_axes=True,
            ticklabels_format=['x', 'y'],
            ticklabels_hide=[],
            ticklabels_wrap=[],
            labels_hide=[],
            **kwargs):
    """
    Create a regression plot using the specified method.

    Parameters
    ----------
    df : pandas DataFrame
        Tidy (“long-form”) dataframe where each column is a variable and each row is an observation without nan.
    method : str, optional
        The correlation method to use. Can be either 'spearman' or 'pearson'. Default is 'spearman'.
    scattersize : int, optional
        The size of the scatter points. Default is 20.
    scattercolor : str, optional
        The color of the scatter points. Default is 'black'.
    linecolor : str, optional
        The color of the regression line. Default is 'red'.
    ax : matplotlib Axes, optional
        The Axes object to draw the plot on. If not provided, a new one will be created.
    figsize : Tuple[int, int], optional
        The size of the figure to create if no Axes object is provided. Default is (3, 3).
    adjust_axes : bool, optional
        Whether to adjust the formatting of the axes. Default is True.
    ticklabels_format : List[str], optional
        Whether to format the tick labels. Default is ['x', 'y'].
    ticklabels_hide : List[str], optional
        Whether to hide the tick labels. Default is [].
    ticklabels_wrap : List[str], optional
        Whether to wrap the tick labels to wrap. Default is [].
    **kwargs
        Additional keyword arguments passed to sns.regplot.

    Returns
    -------
    ax: marplotlib Axes
        The Axes object containing the plot.
    """


    scatter_kws = {'s': scattersize, 'color': scattercolor}
    line_kws = {'color': linecolor}
    df, label, x, y, hue, size = _plotdata_handle(df)

    if size:
        scatter_kws['size'] = df['size']

    if method.lower() == 'spearman':
        corr, pvalue = scipy.stats.spearmanr(df[x], df[y])

    elif method.lower() == 'pearson':
        corr, pvalue = scipy.stats.pearsonr(df[x], df[y])

    else:
        raise KeyError(
            "method parameter should be one of 'spearman' and 'pearson', please check your input parameter."
        )

    methods_name = {'spearman': ' rho: ', 'pearson': ' corr: '}
    title = ' '.join([
        method.title(), methods_name[method],
        '%.2f' % corr, '\nP-value:',
        '%.2e' % pvalue
    ])

    if not ax:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax = sns.regplot(data=df,
                     x=x,
                     y=y,
                     line_kws=line_kws,
                     scatter_kws=scatter_kws,
                     ax=ax,
                     **kwargs)

    if adjust_axes:
        ax = axes_(ax, title=None, ticklabels_format=ticklabels_format,
                   ticklabels_hide=ticklabels_hide, ticklabels_wrap=ticklabels_wrap)
    ax.set_title(title, horizontalalignment='left', loc='left', fontsize=6)

    return ax


@check_type
def heatmap(df,
            lut=None,
            z_score=None,
            nan_policy='omit',
            cmap=RB_CMAP,
            vmax=1,
            vmin=-1,
            center=0,
            col_cluster=False,
            row_cluster=False,
            xticklabels=False,
            yticklabels=True,
            xlabel=None,
            ylabel=None,
            figsize=(4, 4),
            legend=True,
            cbar_pos=(1.6, .35, .03, .18),
            ax=None,
            **kwargs):
    """
    Create a heatmap or clustered heatmap of the given data.

    Parameters
    ----------
    df : pandas DataFrame
        The data to plot.
    lut : Dict, optional
        A nested dictionary mapping column or index names to colors. The columns/index names were the key of the outer dictionary, the components of columns/index were the key of in inner dictionary, the hex code of colors were the value of the inner dictionary.
    z_score : int, optional
        The axis along which to standardize the data. Default is None. 0 for row and 1 for columns.
    nan_policy : str, optional
        How to handle missing values when standard the data. Default is 'omit'.
    cmap : str, optional
        The colormap to use for the heatmap. Default is eplot.constant.RB_CMAP.
    vmax : float, optional
        The maximum value to display in the heatmap. Default is 1.
    vmin : float, optional
        The minimum value to display in the heatmap. Default is -1.
    center : float, optional
        The value at which to center the colormap. Default is 0.
    col_cluster : bool, optional
        Whether to cluster the columns of the heatmap. Default is False.
    row_cluster : bool, optional
        Whether to cluster the rows of the heatmap. Default is False.
    xticklabels : bool, optional
        Whether to display tick labels along the x axis. Default is False.
    yticklabels : bool, optional
        Whether to display tick labels along the y axis. Default is True.
    xlabel : str, optional
        The label for the x axis. Default is None.
    ylabel : str, optional
        The label for the y axis. Default is None.
    figsize : Tuple[int, int], optional
        The size of the figure to create. Default is (4, 4).
    legend : bool, optional
        Whether to display a legend. Default is True.
    **kwargs
        Additional keyword arguments passed to sns.clustermap.

    Returns
    -------
    sns.matrix.ClusterGrid
        The ClusterGrid object representing the heatmap.
    """


    if lut:
        columns_unique = pd.unique(pd.Series(df.columns.names).dropna())
        index_unique = pd.unique(pd.Series(df.index.names).dropna())
        lut_name = np.intersect1d(np.union1d(columns_unique, index_unique), np.asarray(list(set(lut.keys()))))

        if lut_name.size == 0:
            raise ValueError("Pleast check lut parameter.")
        else:
            if set(lut_name) & set(columns_unique):
                order = [i for i in columns_unique if i in lut_name and i == i]
                col_colors = df.columns.to_frame()[order].apply(lambda x: x.map(lut[x.name]))
            else:
                col_colors = None

            if set(lut_name) & set(index_unique):
                order = [i for i in index_unique if i in lut_name]
                row_colors = df.index.to_frame()[order].apply(lambda x: x.map(lut[x.name]))
            else:
                row_colors = None
    else:
        col_colors = None
        row_colors = None

    if any((isinstance(col_colors, pd.DataFrame), isinstance(row_colors, pd.DataFrame), col_cluster, row_cluster)):
        fig = sns.clustermap(df,
            z_score=z_score,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            center=center, col_cluster=col_cluster, row_cluster=row_cluster,
            xticklabels=xticklabels, yticklabels=yticklabels, col_colors=col_colors, row_colors=row_colors, figsize=figsize, cbar_pos=cbar_pos,#legend=legend, 
            **kwargs)
        ax = fig.ax_heatmap

        if legend:
            handle, handle_label = _heatmap_legend_handle(lut)
            ax.legend(handle, handle_label, handler_map={str: LegendTitle({'fontsize': 7})}, bbox_to_anchor=(
                1.05, 0.45), bbox_transform=plt.gcf().transFigure, loc='center left', labelspacing=.3, frameon=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig

    else:
        if z_score in [0, 1]:
            axis = {0: 1, 1: 0}.get(z_score, None)
            df = df.apply(scipy.stats.zscore, axis=axis, nan_policy=nan_policy)
        else:
            vmax = vmax
            vmin = vmin
            center = center
        if not ax:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        ax = sns.heatmap(df,
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            center=center,
            xticklabels=xticklabels, yticklabels=yticklabels,
            ax=ax,
            **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax
    

@check_type
def vennplot(subsets,
             weighted=False,
             labels_name=None,
             palette=None,
             title=None,
             alpha=0.6,
             ax=None,
             subset_label_formatter=None,
             figsize=(3, 3),
             **kwargs):
    if all(isinstance(i, Iterable) for i in subsets):
        for i in range(len(subsets)):
            subsets[i] = set(subsets[i])
    elif any(isinstance(i, Iterable) for i in subsets):
        raise TypeError('Unsupported operand mixed type(s)')
    else:
        pass

    if weighted:
        weight = 'weighted'
    else:
        weight = 'unweighted'

    if not palette:
        palette = MCMAP[:len(subsets)]

    func = {'weighted_2': venn2, 'unweighted_2': venn2_unweighted, 'weighted_3': venn3,
            'unweighted_3': venn3_unweighted}.get(weight+'_'+str(len(subsets)))

    if not ax:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    ax = func(subsets=subsets,
              set_labels=labels_name,
              set_colors=palette,
              alpha=alpha,
              subset_label_formatter=subset_label_formatter,
              ax=ax,
              **kwargs)
    plt.title(title, fontsize=14)
    return ax
