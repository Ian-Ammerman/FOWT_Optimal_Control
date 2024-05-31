import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from Digital_Twin_ZMQ.Prediction_Model.DOLPHINN.vmod.dolphinn import DOLPHINN as DOL
from Digital_Twin_ZMQ.Prediction_Model.prediction_functions import save_prediction_csv, active_pred_plot

# Initialize the figure and axes
plt.ion()
fig, ax = plt.figure(figsize=(10, 6)), plt.axes()

# Define the prediction_history DataFrame with default column names
prediction_history = pd.DataFrame(columns=['Time', 'Predicted_State'])  # DataFrame to store prediction history

def run_DOLPHINN(data_frame_inputs, DOLPHINN_PATH, plot_figure, current_time, pred_error, save_csv, save_csv_time, FOWT_pred_state):
    global prediction_history
    
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

    if plot_figure:
        active_pred_plot(t_pred, y_hat, pred_error, data_frame_inputs, current_time, dol, time_data, t1_idx, t2, t1, fig, ax, FOWT_pred_state)

    # Save data to CSV files when current_time is 1000
    if current_time == save_csv_time and save_csv:
        save_prediction_csv(t_pred, y_hat, pred_error, prediction_history, FOWT_pred_state)

    # Store the prediction history
    history_data = pd.DataFrame({
        'Time': t_pred + t2,
        f'Predicted_{FOWT_pred_state}': y_hat[f"{FOWT_pred_state}"] + pred_error
    })
    history_data.columns = ['Time', f'Predicted_{FOWT_pred_state}']
    prediction_history = pd.concat([prediction_history, history_data])

    return t_pred, y_hat
