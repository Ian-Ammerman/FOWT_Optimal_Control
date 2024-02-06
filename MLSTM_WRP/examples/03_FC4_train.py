import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# DATA
from MLSTM_WRP.vmod import p2v as pp

"""
In this example:
- Perform MLSTM-WRP on Focal Campaign 4 with wind-wave test (data are collected at 24Hz)
"""

TEST_NUM = 2
DATA_INPUT_FILE = r"./MLSTM_WRP/Data/FC4/windwave.csv"
TIME_HORIZON = 20

if not os.path.exists(f"./MLSTM_WRP/examples/figures/{TEST_NUM}"):
    os.makedirs(f"./MLSTM_WRP/examples/figures/{TEST_NUM}")

# Data pre-processing
data = pp.PreProcess(DATA_INPUT_FILE)
data.nan_check()
correlation_matrix = data.idle_sensors_check()
dataset = data.dataset
# Plot the heatmap
mpl.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(7, 7))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.savefig(f".\\figures\\02\\correlation heatmap.pdf", format="pdf")