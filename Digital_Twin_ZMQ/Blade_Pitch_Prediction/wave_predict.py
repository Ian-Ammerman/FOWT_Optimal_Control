import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from DOLPHINN.vmod.dolphinn import DOLPHINN as DOL

# Create a figure and axes outside of your main update function
plt.ion()  # Turn on interactive mode
fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
line_actual, = ax.plot([], [], color='black', label='Measured BlPitch')
line_predicted, = ax.plot([], [], color='red', linestyle='-', label='Predicted BlPitch')
marker_actual = ax.scatter([], [], color='black')
marker_predicted = ax.scatter([], [], color='red')
text_actual = ax.annotate('', xy=(0, 0), xytext=(10,10), textcoords="offset points")
text_predicted = ax.annotate('', xy=(0, 0), xytext=(0,8), textcoords="offset points")

def run_DOLPHINN(data_frame_inputs, DOLPHINN_PATH, update_plot):
    global line_actual, line_predicted, marker_actual, marker_predicted, text_actual, text_predicted
    
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

    state = data[dol.dof].mul(dol.conversion, axis=1).iloc[:t1_idx]
    time_data = data['Time'].iloc[:t2_idx]
    wave = data['wave'].iloc[:t2_idx]

    t_pred, y_hat = dol.predict(time_data, state, wave, history=0)
    
    if update_plot:
        line_actual.set_data(time_data.iloc[0:t1_idx] + t2, state["BlPitchCMeas"][0:t1_idx]*180/np.pi)
        line_predicted.set_data(t_pred + t2, y_hat["BlPitchCMeas"]*180/np.pi)

        # Update marker and text for actual data
        last_actual_time = time_data.iloc[t1_idx-1] + t2
        last_actual_pitch = state["BlPitchCMeas"].iloc[t1_idx-1] * 180/np.pi
        marker_actual.set_offsets((last_actual_time, last_actual_pitch))
        marker_actual.set_label('Current BlPitch')

        # Update marker and text for predicted data
        last_pred_time = t_pred.iloc[-1] + t2
        last_pred_pitch = y_hat["BlPitchCMeas"].iloc[-1] * 180/np.pi
        marker_predicted.set_offsets((last_pred_time, last_pred_pitch))
        marker_predicted.set_label(f'Predicted BlPitch (+{dol.time_horizon}s)')  

        ax.set_xlim((t1 - 50, t1 + 50))
        ax.set_ylim((0, 10))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angle [deg]")
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.draw()
        plt.pause(0.1)

    return t_pred, y_hat
