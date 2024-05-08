import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from vmod.dolphinn import DOLPHINN as DOL

"""
In this example:
- load a trained MLSTM model using DOLPHINN on new FC2 data
"""

# Configure
TEST = "2"
DOLPHINN_PATH = "/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO/DOLPHINN/saved_models/1a/wave_model"
DATA_PATH = "/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO/DOLPHINN/data/S31_10Hz_FS.csv"
PRESENT_TIME = 10000

if not os.path.exists(os.path.join("ZMQ_Prediction_ROSCO", "DOLPHINN", "figures", f"{TEST}")):
    os.makedirs(os.path.join("ZMQ_Prediction_ROSCO", "DOLPHINN", "figures", f"{TEST}"))

# call dolphinn
dol = DOL()
dol.load(DOLPHINN_PATH)

# predict
data = pd.read_csv(DATA_PATH)
t1 = PRESENT_TIME
t2 = dol.time_horizon
t1_idx = np.where(np.min(np.abs(data['Time'] - t1)) == np.abs(data['Time'] - t1))[0][0]
t2_idx = np.where(np.min(np.abs(data['Time']-(t2+t1))) == np.abs(data['Time']-(t2+t1)))[0][0]
state = data[dol.dof].iloc[0:t1_idx]
time = data['Time'].iloc[0:t2_idx]
wave = data['wave'].iloc[0:t2_idx]
t_pred, y_hat = dol.predict(time, state, wave, convert=True, history=500)

plt.figure()
plt.plot(time.iloc[0:t1_idx], state["PtfmRDY"][0:t1_idx], color='black', label='Actual')
plt.plot(t_pred, y_hat["PtfmRDY"], color='red', linestyle='-', label='Predicted')
plt.xlim((t1-250, t1+50))
plt.legend()
plt.savefig(fr"DOLPHINN\.\figures\{TEST}\test.pdf", format="pdf")