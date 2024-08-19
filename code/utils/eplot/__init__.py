import os
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')

from .core import barplot, scatterplot, cateplot, regplot
from .plot_func import bubbleplot, volcanoplot
from .constants import MCMAP, RB_CMAP, R_CMAP

params = {
    'pdf.fonttype': 42,
    'axes.unicode_minus': False,
    'font.sans-serif': 'Arial',
    'lines.linewidth': .5,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.linewidth': .5,
    'xtick.major.width': .5,
    'ytick.major.width': .5,
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'axes.labelsize': 7,
    'axes.titlesize': 8, 
    'font.size': 6
}
mpl.rcParams.update(params)


