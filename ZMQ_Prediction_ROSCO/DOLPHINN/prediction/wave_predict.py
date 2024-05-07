import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from vmod.dolphinn import DOLPHINN as DOL

# Variable to hold the last figure
last_fig = None

def run_DOLPHINN(data_frame_inputs, DOLPHINN_PATH, plot_figure):
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

        if last_fig is not None:
            plt.close(last_fig)

        last_fig = plt.figure(figsize=(10, 6))
        # Plot the actual data
        plt.plot(time_data.iloc[0:t1_idx] + t2, state["BlPitchCMeas"][0:t1_idx]*180/np.pi*180/np.pi, color='black', label='Actual')
        # Plot the predicted data
        plt.plot(t_pred + t2, y_hat["BlPitchCMeas"]*180/np.pi, color='red', linestyle='-', label='Predicted')
        
        # Add marker at the last data point of the actual data series
        last_actual_time = time_data.iloc[t1_idx-1] + t2
        last_actual_pitch = state["BlPitchCMeas"].iloc[t1_idx-1] * 180/np.pi * 180/np.pi
        plt.scatter(last_actual_time, last_actual_pitch, color='blue')  # Blue marker at last actual point
        plt.annotate('Current blade pitch', (last_actual_time, last_actual_pitch), textcoords="offset points", xytext=(10,10), ha='center')
        
        # Add marker at the last data point of the predicted data series
        last_pred_time = t_pred.iloc[-1] + t2
        last_pred_pitch = y_hat["BlPitchCMeas"].iloc[-1] * 180/np.pi
        plt.scatter(last_pred_time, last_pred_pitch, color='green')  # Green marker at last predicted point
        plt.annotate(f'Predicted blade pitch (+{dol.time_horizon}s)', (last_pred_time, last_pred_pitch), textcoords="offset points", xytext=(0,8), ha='center')
        
        plt.xlim((t1 - 50, t1 + 50))
        plt.legend()
        plt.xlabel("Time [s]")  # Set x-axis label
        plt.ylabel("Angle [deg]")  # Set y-axis label
        plt.show()

        # Specify window position
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.wm_geometry("+100+100")  # Position at (x=100, y=100)

        plt.pause(1)

    return t_pred, y_hat
    

