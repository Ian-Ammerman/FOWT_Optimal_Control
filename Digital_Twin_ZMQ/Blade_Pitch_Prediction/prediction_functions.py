import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

def Buffer(Pred_B, t_pred, current_time, measurements, buffer_duration, pred_error, time_horizon):
    # Global variables
    Pred_B_buffer = deque()
    last_used_Pred_B = None  # Initially set to zero
    last_used_t_pred = None  # Initially set to None
    first_delta_received = False
    printed_first_Pred_B = False
    last_whole_second = None  # Track the last whole second for countdown    
    # Buffering Pred_B with its predicted time and the time it was predicted
    if Pred_B is not None:
        Pred_B_buffer.append((Pred_B, current_time + buffer_duration , t_pred))
        if not first_delta_received and not printed_first_Pred_B:
            printed_first_Pred_B = True # Make sure only first Pred_B is printed
            first_delta_received = True  # Set the flag on receiving the first Pred_B
            last_whole_second = int(current_time)  # Initialize countdown start time
            print(f"First pitch angle offset received and buffered: {Pred_B - measurements['BlPitchCMeas']} radians at time {current_time}")

    # Release buffer based on current time and buffer_duration
    while Pred_B_buffer and Pred_B_buffer[0][1] <= current_time:
        last_used_Pred_B, _, last_used_t_pred = Pred_B_buffer.popleft()
        first_delta_received = False  # Reset the flag after the first Pred_B is used

    # Use the last released Pred_B as the control pitch command
    if last_used_Pred_B is not None:
        Pred_Delta_B_0 = last_used_Pred_B * np.pi/180 - measurements['BlPitchCMeas']
        Pred_Delta_B = Pred_Delta_B_0 + pred_error*np.pi/180 # Adding offset in radians
    else:
        Pred_Delta_B = 0.0

    # Countdown for the first Pred_B in the buffer
    if first_delta_received:
        current_whole_second = int(current_time)
        if current_whole_second != last_whole_second:
            last_whole_second = current_whole_second
            if Pred_B_buffer:
                time_to_use = int(Pred_B_buffer[0][1] - current_time)
                print(f"Countdown until first Pred_B prediction is used: {time_to_use} s")

    # Print the current time and prediction time when a Pred_B is used
    if last_used_t_pred is not None and current_time % 1 == 0:
        Time_Buffer = current_time - last_used_t_pred
        Time_Advantage = time_horizon - Time_Buffer
        print(f"Current Time: {current_time}, Last Used Prediction Time: {last_used_t_pred:}, Prediction buffered for: {Time_Buffer:.4f}s (Time advantage: {Time_Advantage:.4f}s)")
        print(f"Sending predicted blade pitch offset setpoint: {Pred_Delta_B:.3f} ({Pred_Delta_B*180/np.pi:.3f} deg)")
    elif last_used_t_pred is None and current_time % 5 == 0:
        print("Blade Pitch Offset Setpoint:", Pred_Delta_B)

    return Pred_Delta_B

# Delta_B saturation to avoid too big prediction offset
def Saturate(Pred_Delta_B, Pred_Saturation, Delta_B_treshold):
    if Pred_Saturation == True:
        if Pred_Delta_B > Delta_B_treshold:
            Pred_Delta_B = Delta_B_treshold
        elif Pred_Delta_B < -Delta_B_treshold:
            Pred_Delta_B = -Delta_B_treshold
    return Pred_Delta_B


def save_prediction_csv(current_time, t_pred, y_hat, pred_error, prediction_history):
    print("Saving results to csv")
    prediction_results_path = os.path.join("Digital_Twin_ZMQ", "Blade_Pitch_Prediction", "prediction_results", "PREDICTION_1000_ACTIVE.csv")
    prediction_history_path = os.path.join("Digital_Twin_ZMQ", "Blade_Pitch_Prediction", "prediction_results", "PRED_HISTORY_1000_ACTIVE.csv")
    
    # Save prediction results
    prediction_results = pd.DataFrame({
        'Time': t_pred + pred_error,
        'Predicted_BlPitchCMeas': y_hat["BlPitchCMeas"] + pred_error
    })
    prediction_results.to_csv(prediction_results_path, index=False)
    
    # Save prediction history
    prediction_history.to_csv(prediction_history_path, index=False)

def active_pred_plot(t_pred, y_hat, pred_error, data_frame_inputs, current_time, dol, time_data, t1_idx, t2):
    # Initialize the figure and axes
    plt.ion()
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    line_actual, = ax.plot([], [], color='blue', label='Measured BlPitch (ROSCO)')
    line_predicted, = ax.plot([], [], color='#3CB371', linestyle='-', label='Predicted BlPitch')
    line_history, = ax.plot([], [], color='#3CB371', linestyle="--", label='Prediction history')  
    marker_actual = ax.scatter([], [], color='blue', alpha=0.5)
    marker_predicted = ax.scatter([], [], color='#3CB371', alpha=0.5)
    old_predictions = []  # List to store old prediction lines
    plotted_times = set()  # Set to store times for which history has been plotted
    last_stippled_time = None  # Track the last time a stippled line was added

    # Convert current predicted line to stippled and add to old predictions if sufficient time has passed
    if last_stippled_time is None or current_time - last_stippled_time >= 10:  # adjust interval as needed
        if line_predicted.get_xdata().size > 0:
            old_line = ax.plot(line_predicted.get_xdata(), line_predicted.get_ydata(), linestyle="--", color='#3CB371')[0]
            old_predictions.append(old_line)
        last_stippled_time = current_time

    # Clear current prediction line data
    line_predicted.set_data([], [])

    # Set new data for current prediction
    line_predicted.set_data(t_pred + t2, y_hat["BlPitchCMeas"] + pred_error)
    line_actual.set_data(time_data.iloc[0:t1_idx] + t2, data_frame_inputs["BlPitchCMeas"].iloc[:t1_idx]*180/np.pi)
    
    # Update marker and text for actual data
    last_actual_time = time_data.iloc[t1_idx-1] + t2
    last_actual_pitch = data_frame_inputs["BlPitchCMeas"].iloc[t1_idx-1] * 180/np.pi
    marker_actual.set_offsets((last_actual_time, last_actual_pitch))
    marker_actual.set_label(f'Current BlPitch ({current_time}s)')

    # Update marker and text for predicted data
    last_pred_time = t_pred.iloc[-1] + t2
    last_pred_pitch = y_hat["BlPitchCMeas"].iloc[-1] + pred_error
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
