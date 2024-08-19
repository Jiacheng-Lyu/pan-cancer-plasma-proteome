import time 
import seaborn as sns 
import matplotlib.colors as mcolors

RB_CMAP = sns.diverging_palette(250, 10, n=30, as_cmap=True)
RDB_CMAP = sns.diverging_palette(250, 10, n=30, as_cmap=True, center='dark')
R_CMAP = sns.light_palette('#D80000', as_cmap=True, n_colors=256)
B_CMAP = sns.light_palette('#005287', as_cmap=True, n_colors=256)
LDR_CMAP = mcolors.LinearSegmentedColormap.from_list("light_dark_red_gradient", ['#F9A2A2', '#ED0000'], N=256)

MCMAP = sns.color_palette("Paired", 30)
TIME_NAME = time.strftime('%Y%m%d', time.localtime(time.time()))
