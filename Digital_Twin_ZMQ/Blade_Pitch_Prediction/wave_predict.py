import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Blade_Pitch_Prediction.DOLPHINN.vmod.dolphinn import DOLPHINN as DOL

# Initialize the figure and axes
plt.ion()
fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
line_actual, = ax.plot([], [], color='blue', label='Measured BlPitch (ROSCO)')
line_predicted, = ax.plot([], [], color='#3CB371', linestyle='-', label=f'Predicted BlPitch')
# Add placeholder for prediction history
line_history, = ax.plot([], [], color='#3CB371', linestyle=(0, (1, 2)), label='Prediction history')  
marker_actual = ax.scatter([], [], color='blue', alpha=0.5)
marker_predicted = ax.scatter([], [], color='#3CB371', alpha=0.5)
old_predictions = []  # List to store old prediction lines
plotted_times = set()  # Set to store times for which history has been plotted
last_stippled_time = None  # Track the last time a stippled line was added

def run_DOLPHINN(data_frame_inputs, DOLPHINN_PATH, update_plot, current_time, prediction_offset):
    global line_actual, line_predicted, marker_actual, marker_predicted, old_predictions, plotted_times, last_stippled_time, line_history
    
    # Load the trained model
    dol = DOL()
    dol.load(DOLPHINN_PATH)

    # Use input data frame directly
    data = data_frame_inputs 
    present_time = round(data_frame_inputs["Time"].iloc[-1] - dol.time_horizon, 4)
    t1 = present_time
    t2 = dol.time_horizon
    t1_idx = np.where(np.min(np.abs(data['Time'] - t1)) == np.abs(data['Time'] - t1))[0][0]
    t2_idx = np.where(np.min(np.abs(data['Time'] - (t2 + t1))) == np.abs(data['Time'] - (t2 + present_time)))[0][0]

    state = data[dol.dof].mul(dol.conversion, axis=1).iloc[:t1_idx]
    time_data = data['Time'].iloc[:t2_idx]
    wave = data['wave'].iloc[:t2_idx]

    t_pred, y_hat = dol.predict(time_data, state, wave, history=0)
    
    if update_plot:
        # Convert current predicted line to stippled and add to old predictions if sufficient time has passed
        if last_stippled_time is None or current_time - last_stippled_time >= 10:  # adjust interval as needed
            if line_predicted.get_xdata().size > 0:
                old_line = ax.plot(line_predicted.get_xdata(), line_predicted.get_ydata(), linestyle=(0, (1, 2)), color='#3CB371')[0]
                old_predictions.append(old_line)
            last_stippled_time = current_time

        # Clear current prediction line data
        line_predicted.set_data([], [])

        # Set new data for current prediction
        line_predicted.set_data(t_pred + t2, y_hat["BlPitchCMeas"]*180/np.pi + prediction_offset)
        line_actual.set_data(time_data.iloc[0:t1_idx] + t2, state["BlPitchCMeas"][0:t1_idx]*180/np.pi)
        
        # Update marker and text for actual data
        last_actual_time = time_data.iloc[t1_idx-1] + t2
        last_actual_pitch = state["BlPitchCMeas"].iloc[t1_idx-1] * 180/np.pi
        marker_actual.set_offsets((last_actual_time, last_actual_pitch))
        marker_actual.set_label(f'Current BlPitch ({current_time}s)')

        # Update marker and text for predicted data
        last_pred_time = t_pred.iloc[-1] + t2
        last_pred_pitch = y_hat["BlPitchCMeas"].iloc[-1] * 180/np.pi + prediction_offset
        marker_predicted.set_offsets((last_pred_time, last_pred_pitch))
        marker_predicted.set_label(f'Predicted BlPitch ({current_time + dol.time_horizon}s)')  

        ax.set_xlim((t1 - 50, t1 + 50))
        ax.set_ylim((0, 10))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angle [deg]")
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.grid(True)
        plt.title(f'Collective Blade Pitch Prediction. Wave Time Horizon: {dol.time_horizon}s')
        plt.draw()
        plt.pause(0.1)

    return t_pred, y_hat
