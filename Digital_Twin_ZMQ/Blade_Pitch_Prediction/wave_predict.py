import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Blade_Pitch_Prediction.DOLPHINN.vmod.dolphinn import DOLPHINN as DOL

# Initialize the figure and axes
plt.ion()
fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
line_actual, = ax.plot([], [], color='blue', label='Measured BlPitch (ROSCO)')
line_predicted, = ax.plot([], [], color='#3CB371', linestyle='-', label=f'Predicted BlPitch')
line_history, = ax.plot([], [], color='#3CB371', linestyle="--", label='Prediction history')  
marker_actual = ax.scatter([], [], color='blue', alpha=0.5)
marker_predicted = ax.scatter([], [], color='#3CB371', alpha=0.5)
old_predictions = []  # List to store old prediction lines
plotted_times = set()  # Set to store times for which history has been plotted
last_stippled_time = None  # Track the last time a stippled line was added

prediction_history = pd.DataFrame(columns=['Time', 'Predicted_BlPitchCMeas'])  # DataFrame to store prediction history

"""
buffer_time_window = 100  # Time window in seconds for the measured buffer
data_buffer_measured = pd.DataFrame(columns=['Time', 'Measured_BlPitch'])  # Buffer to store measured data for the last 100 seconds
data_buffer_predicted = pd.DataFrame(columns=['Time', 'Predicted_BlPitch'])  # Buffer to store predicted data for the last 100 seconds

use_dynamic_offset = False 
fixed_pred_error = 1.4
"""
def run_DOLPHINN(data_frame_inputs, DOLPHINN_PATH, update_plot, current_time, pred_error):
    global line_actual, line_predicted, marker_actual, marker_predicted, old_predictions, plotted_times, last_stippled_time, line_history, prediction_history #, data_buffer_measured, data_buffer_predicted
    
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
    """
    if not y_hat.empty:
        # Update measured buffer with the latest data
        latest_measured = data_frame_inputs["BlPitchCMeas"].iloc[t1_idx-1] * 180/np.pi
        new_data_measured = pd.DataFrame({'Time': [current_time], 'Measured_BlPitch': [latest_measured]})
        data_buffer_measured = pd.concat([data_buffer_measured, new_data_measured], ignore_index=True)
        
        # Update predicted buffer with the latest data, considering the time shift
        latest_predicted = y_hat["BlPitchCMeas"].iloc[-1] + pred_error
        new_data_predicted = pd.DataFrame({'Time': [current_time + t2], 'Predicted_BlPitch': [latest_predicted]})
        data_buffer_predicted = pd.concat([data_buffer_predicted, new_data_predicted], ignore_index=True)
        
        # Remove old data outside the buffer window
        data_buffer_measured = data_buffer_measured[data_buffer_measured['Time'] > current_time - buffer_time_window].reset_index(drop=True)
        data_buffer_predicted = data_buffer_predicted[data_buffer_predicted['Time'] > current_time - buffer_time_window].reset_index(drop=True)
        
        if use_dynamic_offset:
            # Debugging: Print the time values
            print("Measured Times:", data_buffer_measured['Time'].tolist())
            print("Predicted Times:", (data_buffer_predicted['Time'] - t2).tolist())
            
            # Ensure that the data buffers are aligned in time
            aligned_buffer = pd.merge(data_buffer_measured, data_buffer_predicted, left_on='Time', right_on='Time', suffixes=('_Measured', '_Predicted'))
            
            # Calculate pred_error as the mean difference over the adjusted buffer window
            if not aligned_buffer.empty:
                mean_measured = aligned_buffer['Measured_BlPitch'].mean()
                mean_predicted = aligned_buffer['Predicted_BlPitch'].mean()
                pred_error = mean_measured - mean_predicted
                
                # Print the current pred_error
                print(f'Current pred_error: {pred_error:.4f}')
        else:
            pred_error = fixed_pred_error
            
        """   
    if update_plot:
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

    # Save data to CSV files when current_time is 1000
    if current_time == 1000:
        print("Saving results to csv")
        prediction_results_path = "/Users/fredrikfleslandselheim/ROSCO/Digital_Twin_ZMQ/Blade_Pitch_Prediction/prediction_results/PREDICTION_1000_ACTIVE.csv"
        prediction_history_path = "/Users/fredrikfleslandselheim/ROSCO/Digital_Twin_ZMQ/Blade_Pitch_Prediction/prediction_results/PRED_HISTORY_1000_ACTIVE.csv"
        
        # Save prediction results
        prediction_results = pd.DataFrame({
            'Time': t_pred + t2,
            'Predicted_BlPitchCMeas': y_hat["BlPitchCMeas"] + pred_error
        })
        prediction_results.to_csv(prediction_results_path, index=False)
        
        
        # Save prediction history
        prediction_history.to_csv(prediction_history_path, index=False)

    # Store the prediction history
    history_data = pd.DataFrame({
        'Time': t_pred + t2,
        'Predicted_BlPitchCMeas': y_hat["BlPitchCMeas"] + pred_error
    })
    prediction_history = pd.concat([prediction_history, history_data])

    return t_pred, y_hat
