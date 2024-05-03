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
    plt.plot(time_data.iloc[0:t1_idx] + t2, state["BlPitchCMeas"][0:t1_idx], color='black', label='Actual')
    plt.plot(t_pred + t2, y_hat["BlPitchCMeas"], color='red', linestyle='-', label='Predicted')
    plt.xlim((t1-50, t1+50))
    plt.legend()
    plt.show()
 
"""

# Variable to hold the last figure
last_fig = None

def run_DOLPHINN(data_frame_inputs, DOLPHINN_PATH, current_time, plot_figure=False):
    global last_fig
    
    # Load the trained model
    dol = DOL()
    dol.load(DOLPHINN_PATH)

    # Calculate the time span for the prediction
    present_time = round(data_frame_inputs["Time"].iloc[-1] - dol.time_horizon, 4)

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
    
    if plot_figure:
        plt.ion()  # Turn on interactive plotting

        # Close the previous figure if it exists
        if last_fig is not None:
            plt.close(last_fig)

        # Create a new figure and keep track of it
        last_fig = plt.figure()
        plt.plot(time_data.iloc[0:t1_idx] + t2, state["BlPitchCMeas"][0:t1_idx], color='black', label='Actual')
        plt.plot(t_pred + t2, y_hat["BlPitchCMeas"], color='red', linestyle='-', label='Predicted')
        plt.xlim((t1 - 50, t1 + 50))
        plt.legend()
        plt.show()
        plt.pause(0.001)  # Pause for a moment to allow the figure to update

    return t_pred, y_hat
