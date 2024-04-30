import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
print(os.getcwd())
from vmod.dolphinn import DOLPHINN as DOL
"""
In this example:
- load a trained MLSTM model using DOLPHINN on new FC2 data
"""

def run_DOLPHINN():
    print("RUNNING run_prediction")
    # Constants and model paths
    TEST = "2a"
    DOLPHINN_PATH = os.path.join("ZMQ_Prediction_ROSCO", "DOLPHINN", "saved_models", "2a", "wave_model")
    DATA_PATH = os.path.join("ZMQ_Prediction_ROSCO", "DOLPHINN", "data", "test_csv_UPDATED.csv")
    
    # Load the trained model
    dol = DOL()
    dol.load(DOLPHINN_PATH)

    # Calculate the time span for the prediction
    # present_time = current_time - dol.time_horizon
    
    # Use input data frame directly
    #data = data_frame_inputs
    data = pd.read_csv(DATA_PATH)

    present_time = data["Time"].iloc[-1]

    t1 = present_time - 20
    print(t1)
    t2 = dol.time_horizon
    # Find indices for the time interval
    t1_idx = np.where(np.min(np.abs(data['Time'] - t1)) == np.abs(data['Time'] - t1))[0][0]
    t2_idx = np.where(np.min(np.abs(data['Time']-(t2+t1))) == np.abs(data['Time']-(t2+t1)))[0][0]

    # Extract state and wave data up to the relevant time index
    state = data[dol.dof].mul(dol.conversion, axis=1).iloc[:t1_idx]
    time = data['Time'].iloc[:t2_idx]
    wave = data['wave'].iloc[:t2_idx]

    # Perform prediction
    t_pred, y_hat = dol.predict(time, state, wave, history=0)

    plt.figure()
    plt.plot(time.iloc[0:t1_idx], state["PtfmTDZ"][0:t1_idx], color='black', label='Actual')
    plt.plot(t_pred, y_hat["PtfmTDZ"], color='red', linestyle='-', label='Predicted')
    plt.xlim((0, t1+50))
    plt.legend()
    plt.show()

run_DOLPHINN()




