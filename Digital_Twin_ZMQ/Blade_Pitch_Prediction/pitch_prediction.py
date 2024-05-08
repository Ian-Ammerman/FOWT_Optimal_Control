import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from Digital_Twin_ZMQ.Blade_Pitch_Prediction.wave_predict import run_DOLPHINN

class PredictionClass():
    def __init__(self):
        self.port = "5556"
        self.batch_data = [] # To store incoming data temporarily with a fixed max length        
        self.batch_size = 6000  # Number of rows or timesteps per batch
        self.file_generation_count = 0  # Initialize a counter for file generation
        self.t_pred = pd.DataFrame()
        self.y_hat = pd.DataFrame()
        self.has_run_once = False  # Flag to check if simulation has run once
        self.csv_saved = False  # Flag to check if the CSV has been saved once
        self.initial_time = None
        self.timestep = 0.0125
        self.data_frame_inputs = pd.DataFrame()  # Initialize DataFrame
        self.iteration_count = 0  # Initialize the iteration count outside the loop

    def run_simulation(self, current_time, measurements, DOLPHINN_PATH, plot_figure, time_horizon):
        self.iteration_count += 1  # Initialize the iteration count outside the loop

        if not hasattr(self, 'csv_df'):
            print("Retreiving wave data ...")
            csv_file_path = os.path.join("Digital_Twin_ZMQ", "DOLPHINN", "data", "WaveData.csv")
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

        if current_time >= time_horizon + self.initial_time and len(self.batch_data) <= self.batch_size:
            current_index = (current_time - self.initial_time) / self.timestep
            steps_in_horizon = time_horizon / self.timestep
            update_index = round(current_index - steps_in_horizon)
            measurement_values = [measurements.get(key, 0.0) for key in required_measurements]
            self.batch_data[update_index][2:] = measurement_values

        if len(self.batch_data) > self.batch_size:
            steps_in_horizon = time_horizon / self.timestep
            update_index = round(self.batch_size - steps_in_horizon)
            measurement_values = [measurements.get(key, 0.0) for key in required_measurements]
            self.batch_data[update_index][2:] = measurement_values
            popped_row = self.batch_data.pop(0)

        if self.iteration_count % 500 == 0 and len(self.batch_data) < self.batch_size:
            print(f"Remaining rows until initializing DOLPHINN: {self.batch_size - len(self.batch_data)} (Batch size:{len(self.batch_data)})")

        # Check if the last time value is at a whole second
        if len(self.batch_data) >= self.batch_size and current_time % 1 == 0:
            data_frame_inputs = pd.DataFrame(self.batch_data, columns=['Time', 'wave'] + required_measurements)
            print("Running DOLPHINN with input data frame shape:", data_frame_inputs.shape)
            self.t_pred, self.y_hat = run_DOLPHINN(data_frame_inputs, DOLPHINN_PATH, plot_figure)
            # print("Predicted Collective Blade Pitch Angle:", self.y_hat["BlPitchCMeas"].iloc[-1])
            # print("t_pred:", self.t_pred.iloc[-1])
                        
            if self.csv_saved is False and current_time % 200 == 0:
                self.csv_saved = True
                output_file_path = os.path.join("Digital_Twin_ZMQ", "Blade_Pitch_Prediction", "Control_Data", f"Control_Data_T{current_time}.csv")
                data_frame_inputs.to_csv(output_file_path, index=False)
                print(f"SAVED control CSV at t = {current_time}")

        if hasattr(self, 'y_hat') and not self.y_hat.empty:
            return self.y_hat["BlPitchCMeas"].iloc[-1], self.t_pred.iloc[-1]
        else:
            return None, None
