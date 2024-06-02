import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

class PredictionClass():
    def __init__(self):
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.port = "5556"
        self.batch_data = []  # To store incoming data temporarily with a fixed max length        
        self.batch_size = 6000  # Number of rows or timesteps per batch
        self.file_generation_count = 0  # Initialize a counter for file generation
        self.t_pred = pd.DataFrame()
        self.y_hat = pd.DataFrame()
        self.present_state = None
        self.has_run_once = False  # Flag to check if simulation has run once
        self.csv_saved = False  # Flag to check if the CSV has been saved once
        self.initial_time = None
        self.timestep = 0.0125
        self.data_frame_inputs = pd.DataFrame()  # Initialize DataFrame
        self.iteration_count = 0  # Initialize the iteration count outside the loop
        self.full_measurements = []  # To store all measurements for all timesteps

    def run_simulation(self, current_time, measurements, plot_figure, time_horizon, pred_error, pred_freq, save_csv, save_csv_time, FUTURE_WAVE_FILE, FOWT_pred_state, MLSTM_MODEL_NAME):
        from Digital_Twin_ZMQ.Prediction_Model.wave_predict import run_DOLPHINN
        DOLPHINN_PATH  = os.path.join(self.this_dir, "DOLPHINN", "saved_models", MLSTM_MODEL_NAME, "wave_model")

        self.iteration_count += 1  # Initialize the iteration count outside the loop

        if not hasattr(self, 'csv_df'):
            print("Retrieving incoming wave data ...")
            csv_file_path = os.path.join(self.this_dir, "Incoming_Waves", FUTURE_WAVE_FILE)
            self.csv_df = pd.read_csv(csv_file_path)

        required_measurements = ['PtfmTDX', 'PtfmTDZ', 'PtfmTDY', 'PtfmRDX', 'PtfmRDY', 'PtfmRDZ', 'BlPitchCMeas', 'RotSpeed']

        # Set initial time if it has not been set
        if self.initial_time is None:
            self.initial_time = current_time
        desired_time = current_time + time_horizon

        matching_rows = self.csv_df[np.isclose(self.csv_df['Time'], desired_time)]
        wave_measurement = matching_rows['wave'].iloc[0]
        # For all timesteps, append Time and wave
        self.batch_data.append([current_time, wave_measurement] + [None] * len(required_measurements))
        self.full_measurements.append([current_time] + [measurements.get(key, 0.0) for key in required_measurements])  # Store all measurements

        if current_time >= time_horizon + self.initial_time and len(self.batch_data) <= self.batch_size:
            future_index = (current_time - self.initial_time) / self.timestep
            steps_in_horizon = time_horizon / self.timestep
            update_index = round(future_index - steps_in_horizon)
            measurement_values = [measurements.get(key, 0.0) for key in required_measurements]
            self.batch_data[update_index][2:] = measurement_values

        if len(self.batch_data) > self.batch_size:
            steps_in_horizon = time_horizon / self.timestep
            update_index = round(self.batch_size - steps_in_horizon)
            measurement_values = [measurements.get(key, 0.0) for key in required_measurements]
            self.batch_data[update_index][2:] = measurement_values
            popped_row = self.batch_data.pop(0)

        if self.iteration_count % 200 == 0 and len(self.batch_data) < self.batch_size:
            print(f"Remaining rows until initializing DOLPHINN: {self.batch_size - len(self.batch_data)} (Batch size: {len(self.batch_data)})")
            
        # Check if the last time value is at a whole second
        if len(self.batch_data) >= self.batch_size and current_time % pred_freq == 0:
            data_frame_inputs = pd.DataFrame(self.batch_data, columns=['Time', 'wave'] + required_measurements)
            print("Running DOLPHINN with input data frame shape:", data_frame_inputs.shape)
            self.t_pred, self.y_hat = run_DOLPHINN(data_frame_inputs, DOLPHINN_PATH, plot_figure, current_time, pred_error, save_csv, save_csv_time, FOWT_pred_state)
            if not self.csv_saved and current_time == 1000:
                self.control_csv_saved = True
                # Save only the required_measurements along with the Time column
                output_file_path = os.path.join(self.this_dir, "Control_Data", f"Control_Data_T{current_time}_active.csv")
                data_frame_inputs.to_csv(output_file_path, index=False)
                print(f"SAVED control CSV at t = {current_time}")

            if current_time == save_csv_time and save_csv:
                full_measurements_df = pd.DataFrame(self.full_measurements, columns=['Time'] + required_measurements)
                full_output_file_path = os.path.join(self.this_dir, "prediction_results", f"measurements_{current_time}_ACTIVE.csv")
                full_measurements_df.to_csv(full_output_file_path, index=False)
                print(f"SAVED measurements at t = {current_time}")

            self.data_frame_inputs = data_frame_inputs  # Ensure data_frame_inputs is assigned

        if hasattr(self, 'y_hat') and not self.y_hat.empty:
            return self.y_hat, self.t_pred
        else:
            return None, None
