import numpy as np

from .base import volcano_category, check_type, axes_
from .core import barplot, scatterplot
import matplotlib.colors as mcolors

cmap = mcolors.LinearSegmentedColormap.from_list(
    "red_blue_gradient", ['#1c3e9f', 'white', '#fc0000'], N=256)

def bubbleplot(df, palette=cmap, **kwargs):
    ax = scatterplot(df,
                     palette=palette,
                     spines_hide=[],
                     ticklabels_hide=[],
                     ticklabels_format=[],
                     **kwargs)
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] * 0.8, xlim[1] * 1.25)

    return ax

def volcanoplot(df,
                title=None,
                highlight_points=[],
                sig_up_color='#D9590A',
                sig_down_color='#8875C7',
                nosig_color='lightgray',
                pvalue_cutoff=0.05,
                ratio_cutoff=2,
                highlight_ratio=3,
                ticklabels_hide=[],
                ticklabels_format=['x', 'y'],
                spines_hide=[],
                text_label=None,
                legend='brief',
                return_data=False,
                **kwargs):
    df = df.copy(deep=True)
    if df.iloc[:, 0].min() > 0:
        df.iloc[:, 0] = np.log2(df.iloc[:, 0].astype(float))
    if df.iloc[:, 1].max() <= 1:
        df.iloc[:, 1] = -np.log10(df.iloc[:, 1].astype(float))

    df.columns = [
        '$\mathregular{Log_2}$(Fold change)',
        '-$\mathregular{Log_{10}}$(FDR)' if ('adjust' or 'FDR')
        in df.columns[1] else '-$\mathregular{Log_{10}}$(p-value)'
    ]

    df[['type', 'size']] = df.apply(lambda x: volcano_category(
        x.iloc[0], x.iloc[1], pvalue_cutoff, ratio_cutoff),
        axis=1,
        result_type='expand')

    df = df.sort_values(by='type', ascending=True)
    palette = dict(zip(['Downregulated', 'NS', 'Upregulated'], [sig_down_color, nosig_color, sig_up_color]))

    if highlight_points:
        df.loc[highlight_points,
               'size'] = highlight_ratio * df.loc[highlight_points, 'size']
    if not title:
        title = 'Volcanoplot'
    ax = scatterplot(df,
                     title,
                     palette=palette,
                     highlight_points=highlight_points,
                     ticklabels_hide=ticklabels_hide,
                     ticklabels_format=ticklabels_format,
                     spines_hide=spines_hide,
                     text_label=text_label,
                     legend=legend,
                     **kwargs)

    ax.axhline(y=-np.log10(pvalue_cutoff),
               c="black",
               ls="--",
               dashes=(11, 8),
               lw=0.5)

    for value in [-np.log2(ratio_cutoff), np.log2(ratio_cutoff)]:
        ax.axvline(x=value, c="black", ls="--", dashes=(11, 8), lw=0.5)
    if return_data:
        return df.iloc[:, :2], ax
    else:
        return ax

@check_type
def stack_barplot(dfs,
                  palettes=None,
                  title=None,
                  ax=None,
                  orient='v',
                  figsize=(3, 3),
                  adjust_axes=True,
                  ticklabels_hide=['x'],
                  ticklabels_format=['y'],
                  spines_hide=['top', 'right'],
                  labels_hide=None,
                  linewidth=0,
                  **kwargs):
    if palettes:
        if len(palettes) != len(dfs):
            raise ValueError(
                'The count of palettes must be consistent with the dfs!')
    else:
        palettes = [None] * len(dfs)

    for palette, df in zip(palettes, dfs):
        ax = barplot(df,
                     palette=palette,
                     title=title,
                     orient=orient,
                     ax=ax,
                     figsize=figsize,
                     linewidth=linewidth,
                     adjust_axes=False,
                     **kwargs)

    if adjust_axes:
        ax = axes_(ax,
                   title,
                   ticklabels_hide=ticklabels_hide,
                   ticklabels_format=ticklabels_format,
                   spines_hide=spines_hide,
                   labels_hide=labels_hide)

    return ax