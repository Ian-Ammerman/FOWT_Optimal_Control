import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from ZMQ_Prediction_ROSCO.DOLPHINN.vmod.dolphinn import DOLPHINN as DOL

"""
In this example:
- load a trained MLSTM model using DOLPHINN on new FC2 data
"""

def run_DOLPHINN():
    print("RUNNING run_prediction")
    # Constants and model paths
    TEST = "2a"
    DOLPHINN_PATH = "/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO/DOLPHINN/saved_models/2a/wave_model"
    DATA_PATH = "/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO/DOLPHINN/data/data_frame_inputs.csv"
    
    # Load the trained model
    dol = DOL()
    dol.load(DOLPHINN_PATH)

    # Calculate the time span for the prediction
    # present_time = current_time - dol.time_horizon
    present_time = 54.9875
    # Use input data frame directly
    # data = data_frame_inputs
    data = pd.read_csv(DATA_PATH)
    print(data.shape)

    # Find indices for the time interval
    t1_idx = np.where(np.min(np.abs(data['Time'] - present_time)) == np.abs(data['Time'] - present_time))[0][0]
    t2_idx = np.where(np.min(np.abs(data['Time'] - (dol.time_horizon + present_time))) == np.abs(data['Time'] - (dol.time_horizon + present_time)))[0][0]

    # Extract state and wave data up to the relevant time index
    state = data[dol.dof].mul(dol.conversion, axis=1).iloc[:t1_idx]
    time = data['Time'].iloc[:t2_idx]
    wave = data['wave'].iloc[:t2_idx]

    # Perform prediction
    t_pred, y_hat = dol.predict(time, state, wave, history=5000)

    return t1_idx, present_time, state, t_pred, y_hat
