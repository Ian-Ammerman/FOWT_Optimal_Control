import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Buffer prediction setpoint and sending at optimized time. Only used for FOWT_pred_state = BlPitchCMeas
def Buffer(Pred_B, t_pred, current_time, measurements, buffer_duration, pred_error, time_horizon, Prediction):
    # Initialize function attributes if they don't exist
    if not hasattr(Buffer, 'Pred_B_buffer'):
        Buffer.Pred_B_buffer = deque()
        Buffer.last_used_Pred_B = None
        Buffer.last_used_t_pred = None
        Buffer.first_delta_received = False
        Buffer.printed_first_Pred_B = False
        Buffer.last_whole_second = None
    
    # Buffering Pred_B with its predicted time and the time it was predicted
    if Pred_B is not None:
        Buffer.Pred_B_buffer.append((Pred_B, current_time + buffer_duration , t_pred))
        if not Buffer.first_delta_received and not Buffer.printed_first_Pred_B:
            Buffer.printed_first_Pred_B = True # Make sure only first Pred_B is printed
            Buffer.first_delta_received = True  # Set the flag on receiving the first Pred_B
            Buffer.last_whole_second = int(current_time)  # Initialize countdown start time
            print(f"First pitch angle offset received and buffered: {Pred_B - measurements['BlPitchCMeas']} radians at time {current_time}")

    # Release buffer based on current time and buffer_duration
    while Buffer.Pred_B_buffer and Buffer.Pred_B_buffer[0][1] <= current_time:
        Buffer.last_used_Pred_B, _, Buffer.last_used_t_pred = Buffer.Pred_B_buffer.popleft()
        Buffer.first_delta_received = False  # Reset the flag after the first Pred_B is used

    # Use the last released Pred_B as the control pitch command
    if Buffer.last_used_Pred_B is not None:
        Pred_Delta_B_0 = Buffer.last_used_Pred_B * np.pi/180 - measurements['BlPitchCMeas']
        Pred_Delta_B = Pred_Delta_B_0 + pred_error*np.pi/180 # Adding offset in radians
        Pred_B_Buffered = Buffer.last_used_Pred_B + pred_error
    else:
        Pred_Delta_B = 0.0
        Pred_B_Buffered = 0.0

    # Countdown for the first Pred_B in the buffer
    if Buffer.first_delta_received:
        current_whole_second = int(current_time)
        if current_whole_second != Buffer.last_whole_second:
            Buffer.last_whole_second = current_whole_second
            if Buffer.Pred_B_buffer:
                time_to_use = int(Buffer.Pred_B_buffer[0][1] - current_time)
                print(f"Countdown until first Pred_B prediction is used: {time_to_use} s")

    # Print the current time and prediction time when a Pred_B is used
    if Buffer.last_used_t_pred is not None and current_time % 1 == 0:
        Time_Buffer = current_time - Buffer.last_used_t_pred
        Time_Advantage = time_horizon - Time_Buffer
        print(f"Current Time: {current_time}, Last Used Prediction Time: {Buffer.last_used_t_pred}, Prediction buffered for: {Time_Buffer:.4f}s (Time advantage: {Time_Advantage:.4f}s)")
        if Prediction:
            print(f"Sending predicted blade pitch offset setpoint: {Pred_Delta_B:.3f} ({Pred_Delta_B*180/np.pi:.3f} deg)")
    elif Buffer.last_used_t_pred is None and current_time % 5 == 0:
        print("Blade Pitch Offset Setpoint:", Pred_Delta_B)

    return Pred_Delta_B, Pred_B_Buffered


# Delta_B saturation to avoid too big prediction offset
def Saturate(Pred_Delta_B, Pred_Saturation, Delta_B_treshold):
    if Pred_Saturation == True:
        if Pred_Delta_B > Delta_B_treshold:
            Pred_Delta_B = Delta_B_treshold
        elif Pred_Delta_B < -Delta_B_treshold:
            Pred_Delta_B = -Delta_B_treshold
    return Pred_Delta_B

def save_prediction_csv(t_pred, y_hat, pred_error, prediction_history, FOWT_pred_state):
    print("Saving results to csv")
    prediction_results_path = os.path.join("Digital_Twin_ZMQ", "Prediction_Model", "prediction_results", "PREDICTION_1000_ACTIVE.csv")
    prediction_history_path = os.path.join("Digital_Twin_ZMQ", "Prediction_Model", "prediction_results", "PRED_HISTORY_1000_ACTIVE.csv")
    
    # Save prediction results
    prediction_results = pd.DataFrame({
        'Time': t_pred + pred_error,
        f'Predicted_{FOWT_pred_state}': y_hat[f"{FOWT_pred_state}"] + pred_error
    })
    prediction_results.to_csv(prediction_results_path, index=False)
    
    # Save prediction history
    prediction_history.to_csv(prediction_history_path, index=False)

def active_pred_plot(t_pred, y_hat, pred_error, data_frame_inputs, current_time, dol, time_data, t1_idx, t2, t1, fig, ax, FOWT_pred_state):
    # Initialize or update the plot elements
    if not hasattr(active_pred_plot, 'initialized'):
        active_pred_plot.line_actual, = ax.plot([], [], color='blue', label=f'Measured {FOWT_pred_state} (ROSCO)')
        active_pred_plot.line_predicted, = ax.plot([], [], color='#3CB371', linestyle='-', label=f'Predicted {FOWT_pred_state}')
        active_pred_plot.line_history, = ax.plot([], [], color='#3CB371', linestyle="--", label='Prediction history')
        active_pred_plot.marker_actual = ax.scatter([], [], color='blue', alpha=0.5)
        active_pred_plot.marker_predicted = ax.scatter([], [], color='#3CB371', alpha=0.5)
        active_pred_plot.old_predictions = []  # List to store old prediction lines
        active_pred_plot.plotted_times = set()  # Set to store times for which history has been plotted
        active_pred_plot.last_stippled_time = None  # Track the last time a stippled line was added
        active_pred_plot.initialized = True

    # Convert current predicted line to stippled and add to old predictions if sufficient time has passed
    if active_pred_plot.last_stippled_time is None or current_time - active_pred_plot.last_stippled_time >= 10:  # adjust interval as needed
        if active_pred_plot.line_predicted.get_xdata().size > 0:
            old_line = ax.plot(active_pred_plot.line_predicted.get_xdata(), active_pred_plot.line_predicted.get_ydata(), linestyle="--", color='#3CB371')[0]
            active_pred_plot.old_predictions.append(old_line)
        active_pred_plot.last_stippled_time = current_time

    # Clear current prediction line data
    active_pred_plot.line_predicted.set_data([], [])

    # Set new data for current prediction
    active_pred_plot.line_predicted.set_data(t_pred + t2, y_hat[f"{FOWT_pred_state}"] + pred_error)
    active_pred_plot.line_actual.set_data(time_data.iloc[0:t1_idx] + t2, data_frame_inputs[f"{FOWT_pred_state}"].iloc[:t1_idx]*180/np.pi)
    
    # Update marker and text for actual data
    last_actual_time = time_data.iloc[t1_idx-1] + t2
    last_actual_state = data_frame_inputs[f"{FOWT_pred_state}"].iloc[t1_idx-1] * 180/np.pi
    active_pred_plot.marker_actual.set_offsets((last_actual_time, last_actual_state))
    active_pred_plot.marker_actual.set_label(f'Current {FOWT_pred_state} ({current_time}s)')

    # Update marker and text for predicted data
    last_pred_time = t_pred.iloc[-1] + t2
    last_pred_state = y_hat[f"{FOWT_pred_state}"].iloc[-1] + pred_error
    active_pred_plot.marker_predicted.set_offsets((last_pred_time, last_pred_state))
    active_pred_plot.marker_predicted.set_label(f'Predicted {FOWT_pred_state} ({current_time + dol.time_horizon}s)')  

    # Determine y_label based on the FOWT_pred_state value
    if FOWT_pred_state in ["BlPitchCMeas", "PtfmRDX", "PtfmRDY", "PtfmRDZ"]:
        y_label = "Angle [deg]"
    else:
        y_label = "[m]"

    ax.set_xlim((t1 - 50, t1 + 50))
    ax.set_ylim((0, 10))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(y_label)
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.grid(True)
    plt.title(f'{FOWT_pred_state} Prediction. Wave Time Horizon: {dol.time_horizon}s')
    plt.draw()
    plt.pause(0.1)
