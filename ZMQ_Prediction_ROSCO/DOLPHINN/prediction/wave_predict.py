import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from vmod.dolphinn import DOLPHINN as DOL

"""
In this example:
- load a trained MLSTM model using DOLPHINN on new FC2 data

Example for plotting for each iteration:
    plt.figure()
    plt.plot(time_data.iloc[0:t1_idx] + t2, state["PtfmTDZ"][0:t1_idx], color='black', label='Actual')
    plt.plot(t_pred + t2, y_hat["PtfmTDZ"], color='red', linestyle='-', label='Predicted')
    plt.xlim((t1-50, t1+50))
    plt.legend()
    plt.show()

"""

def run_DOLPHINN(data_frame_inputs, current_time):
    # Constants and model paths
    TEST = "2a"
    DOLPHINN_PATH = os.path.join("ZMQ_Prediction_ROSCO", "DOLPHINN", "saved_models", "2a", "wave_model")
    DATA_PATH = os.path.join("ZMQ_Prediction_ROSCO", "DOLPHINN", "data", "data_frame_inputs.csv")
    
    # Load the trained model
    dol = DOL()
    dol.load(DOLPHINN_PATH)

    # Calculate the time span for the prediction
    present_time = round(data_frame_inputs["Time"].iloc[-1] - dol.time_horizon, 4)

    # present_time = 54.9875
    # Use input data frame directly
    data = data_frame_inputs

    t1 = present_time
    t2 = dol.time_horizon
    t1_idx = np.where(np.min(np.abs(data['Time'] - t1)) == np.abs(data['Time'] - t1))[0][0]
    t2_idx = np.where(np.min(np.abs(data['Time'] - (t2 + t1))) == np.abs(data['Time'] - (t2 + present_time)))[0][0]

    # Extract state and wave data up to the relevant time index
    state = data[dol.dof].mul(dol.conversion, axis=1).iloc[:t1_idx]
    time_data = data['Time'].iloc[:t2_idx]
    wave = data['wave'].iloc[:t2_idx]

    # Perform prediction
    t_pred, y_hat = dol.predict(time_data, state, wave, history=0)

    return t_pred, y_hat
