import os

import sys
sys.path.append('.')
import time

import requests
import glob

import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from .main import Dataset, Group, Correlation, Analysis
from .ml_utils import ML_model
from .function import dropnan, ora, r_func

from .eplot.core import barplot, scatterplot, lineplot, cateplot, regplot, heatmap, vennplot
from .eplot.plot_func import bubbleplot, volcanoplot, stack_barplot
from .eplot.constants import RB_CMAP, R_CMAP, MCMAP, LDR_CMAP, B_CMAP

np.set_printoptions(suppress=True)