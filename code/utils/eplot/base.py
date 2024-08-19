import os 
from collections import Iterable

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as transforms

from textwrap import wrap
from statannotations.Annotator import Annotator
from adjustText import adjust_text

from utils.exceptions import ParameterError
from .constants import TIME_NAME


def format_zero_func(x, s='0f'):
    '''
    change ax ticklables 0.0 to 0
    method: ax.xaxis.set_major_formatter(plt.FuncFormatter(format_zero_func))
    '''

    if x == 0:
        return '0'
    else:
        return '%.{}'.format(s) % int(x)


def handle_legend(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
    return ax


def axes_(ax,
          title,
          ticklabels_hide=None,
          ticklabels_format=None,
          ticklabels_wrap=['y'],
          wrap_length=50,
          spines_hide=None,
          labels_hide=None,
          legend=True):
    if title:
        ax.set_title(title)

    if spines_hide and isinstance(spines_hide, (list, set)):
        for side in spines_hide:
            ax.spines[side].set_visible(False)

    ax.tick_params(axis='both')
    if ticklabels_hide:
        if 'x' in ticklabels_hide:
            ax.set_xticklabels('')
            ax.set_xticks([])
        if 'y' in ticklabels_hide:
            ax.set_yticklabels('')
            ax.set_yticks([])
    else:
        if ticklabels_format:
            tick_format = mticker.FuncFormatter(format_zero_func)
            sci_formatter = mticker.ScalarFormatter(useMathText=True)
            sci_formatter.set_scientific(True)
            sci_formatter.set_powerlimits((0, 3))

            if 'x' in ticklabels_format:
                ax.xaxis.set_major_formatter(tick_format)
                ax.xaxis.set_major_formatter(sci_formatter)
                
            if 'y' in ticklabels_format:
                ax.yaxis.set_major_formatter(tick_format)
                ax.yaxis.set_major_formatter(sci_formatter)

    if ticklabels_wrap:
        def wrap_func(x): return '\n'.join(wrap(x.get_text(), wrap_length))
        plt.draw()

        if 'x' in ticklabels_wrap:
            labels = ax.get_xticklabels()
            ticks = ax.get_xticks()
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
            ax.set_xticklabels(list(map(wrap_func, labels)))
        if 'y' in ticklabels_wrap:
            labels = ax.get_yticklabels()
            ticks = ax.get_yticks()
            ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
            ax.set_yticklabels(list(map(wrap_func, labels)))

    if labels_hide:
        if 'x' in labels_hide:
            ax.set_xlabel('')
        if 'y' in labels_hide:
            ax.set_ylabel('')
    if legend:
        handle_legend(ax)

    return ax


def check_type(func):
    def wrapper(df, *args, **kwargs):
        if func.__name__ in ['stack_barplot', 'vennplot']:
            obj_types = {'stack_barplot': 'pandas ', 'vennplot': ''}
            if not isinstance(df, Iterable):
                raise TypeError('first argument must be an iterable of {}objects in {}, you passed an object of type "{}"'.format(
                    obj_types[func.__name__], func.__name__, type(df)))
            else:
                pass
        elif func.__name__ not in ['heatmap', 'clustermap']:
            if isinstance(df, pd.Series):
                if not df.index.name:
                    df.index.name = 'x'
                if not df.name:
                    df.name = 'y'
            elif isinstance(df, pd.DataFrame):
                pass
            else:
                raise TypeError(
                    'The {} data should be a Series or a Dataframe, please check your input df parameter.'
                    .format(func.__name__))
            df = df.reset_index()

        else:
            if isinstance(df, pd.DataFrame):
                pass
            else:
                raise TypeError(
                    'The {} data should be a Dataframe, please check your input df parameter.'
                    .format(func.__name__))
        return func(df, *args, **kwargs)

    return wrapper


def prob_star(pvalue):
    if pvalue > 0.05:
        return 'na'
    elif pvalue > 0.01:
        return '*'
    elif pvalue > 0.001:
        return '**'
    elif pvalue > 1e-4:
        return '***'
    else:
        return '****'


def add_stats(ax, df, x, y, 
              order=None, 
              hue=None, 
              hue_order=None, 
              box_pairs=None, 
              plot='boxplot',
              probs=None, 
              test_short_name=None, 
              loc='inside', 
              test=None,
              line_offset=0.2,
              line_height=0,
              text_offset=4,
              line_width=0.5,
              line_offset_to_group=0.2,
              fontsize=6,
              verbose=0,
              text_format='star',
              orient='v',
              *args,
              **kwargs):
    annot = Annotator(ax, box_pairs, data=df, x=x, y=y, order=order, hue=hue, hue_order=hue_order, orient=orient, plot=plot)
    annot.configure(test=test, test_short_name=test_short_name, loc=loc, verbose=verbose, line_offset=line_offset, line_height=line_height, line_width=line_width, text_offset=text_offset, fontsize=fontsize, text_format=text_format)
    if probs:
        annot.set_pvalues(pvalues=probs)
    else:
        annot.apply_test()
    annot.annotate(line_offset_to_group=line_offset_to_group)
    return ax


def volcano_category(f, p, pvalue_cutoff=0.05, ratio_cutoff=2):
    pv = -np.log10(pvalue_cutoff)
    fc = np.log2(ratio_cutoff)
    if p > pv and f > fc:
        return 'Upregulated', 8
    elif p > pv and f < -fc:
        return 'Downregulated', 8
    else:
        return 'NS', 2


def adjusttext(df, ax, highlight_points, x, y):
    texts = [
        ax.text(df.loc[i, x],
                 df.loc[i, y],
                 i.split('_', 1)[-1].replace('_', ' '),
                 fontsize=6) for i in highlight_points if i in df.index
    ]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='black'))


def _heatmap_legend_handle(lut):
    handles = []
    handles_name = []
    for type_ in lut:
        i = 0
        for name in lut[type_]:
            if i == 0:
                handles.extend(['', type_])
                handles_name.extend(['', ''])
            handles.append(Patch(facecolor=lut[type_][name]))
            handles_name.append(name)
            i += 1
    return handles, handles_name


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    xlim = (mean_x - scale_x, mean_x + scale_x)
    
    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    y_lim = (mean_x - scale_y, mean_x + scale_y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    return xlim, y_lim


def _plotdata_handle(df, palette=None):
    if isinstance(palette, str):
        palette = [palette]

    if len(df.columns) == 2:
        x, y = df.columns
        label, hue, size = None, None, None

    elif len(df.columns) == 3:
        label, x, y = df.columns
        hue, size = None, None

    elif len(df.columns) == 4:
        label, x, hue, y = df.columns
        size = None

    elif len(df.columns) == 5:
        label, x, y, hue, size = df.columns

    else:
        raise ParameterError(
            'The plot data should only contain x, y, group(hue) column')

    if palette and hue and df[hue].nunique() != len(palette):
        raise ParameterError(
            "the palettes' number must be equal to hue groups.")

    return df, label, x, y, hue, size


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle,
                           usetex=False, **self.text_props)
        handlebox.add_artist(title)
        return title


def savefig(ax, outpath=None, outname=None, plot_type=None, out_format='.pdf'):
    if not outpath:
        outpath = os.path.join(os.getcwd(), 'figure')
        if not os.path.isdir(outpath):
            os.mkdir(outpath)

    if isinstance(out_format, str):
        out_format = [out_format]

    out_file_paths = [os.path.join(outpath, '_'.join([outname.replace(
        ' ', '_'), plot_type, TIME_NAME]) + '.' + i.split('.')[-1]) for i in out_format]
    for out_file_path in out_file_paths:
        ax.get_figure().savefig(out_file_path, bbox_inches='tight')
