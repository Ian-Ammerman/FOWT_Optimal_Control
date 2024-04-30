import zmq
import json
import sys
import time
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque

# Dynamically add the grandparent of the grandparent of the current script directory to sys.path
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Importing modules from a relative path
from ZMQ_Prediction_ROSCO.DOLPHINN.vmod.dolphinn import DOLPHINN as DOL
from ZMQ_Prediction_ROSCO.DOLPHINN.prediction.wave_predict import run_DOLPHINN
print("RUNNING PITCH_PREDICTION.PY from prediction")

# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class PredictionClass():
    def __init__(self):
        self.port = "5556"
        self.publisher = self.setup_zmq_publisher()
        self.batch_data = [] # To store incoming data temporarily with a fixed max length        
        self.batch_size = 6000  # Number of rows or timesteps per batch
        self.file_generation_count = 0  # Initialize a counter for file generation
        self.t_preds = None
        self.y_preds = None
        self.has_run_once = False  # Flag to check if simulation has run once
        self.csv_saved = False  # Flag to check if the CSV has been saved once
        self.initial_time = None
        self.time_horizon = 20
        self.timestep = 0.0125
        self.data_frame_inputs = pd.DataFrame()  # Initialize DataFrame
        self.iteration_count = 0  # Initialize the iteration count outside the loop

    def run_simulation(self, current_time, measurements):
        self.iteration_count += 1  # Initialize the iteration count outside the loop

        if not hasattr(self, 'csv_df'):
            print("Retreiving wave data ...")
            csv_file_path = os.path.join("ZMQ_Prediction_ROSCO", "DOLPHINN", "data", "WaveData.csv")
            self.csv_df = pd.read_csv(csv_file_path)
        
        required_measurements = ['PtfmTDX', 'PtfmTDY', 'PtfmTDZ', 'PtfmRDX', 'PtfmRDY', 'PtfmRDZ', 'FA_Acc']

        # Set initial time if it has not been set
        if self.initial_time is None:
            self.initial_time = current_time
            print("Initial time:", self.initial_time)

        wave_measurement = self.csv_df.loc[self.csv_df['Time'] == current_time, 'wave'].iloc[0]

         # For all timesteps, append Time and wave
        self.batch_data.append([current_time, wave_measurement] + [None] * len(required_measurements))

        if current_time >= self.time_horizon + self.initial_time and len(self.batch_data) <= self.batch_size:
            current_index = (current_time - self.initial_time) / self.timestep
            steps_in_horizon = self.time_horizon / self.timestep
            update_index = round(current_index - steps_in_horizon)
            # Ensure update_index is within the current batch data range
            measurement_values = [measurements.get(key, 0.0) for key in required_measurements]
            self.batch_data[update_index][2:] = measurement_values

        if len(self.batch_data) > self.batch_size:
            steps_in_horizon = self.time_horizon / self.timestep
            update_index = round(self.batch_size - steps_in_horizon)
            # Ensure update_index is within the current batch data range
            measurement_values = [measurements.get(key, 0.0) for key in required_measurements]
            self.batch_data[update_index][2:] = measurement_values
            popped_row = self.batch_data.pop(0)
            # Print the row that was popped
            # print("Popped row:", popped_row)

        if self.iteration_count % 100 == 0 and len(self.batch_data) < self.batch_size:
            print(f"Number of rows: {len(self.batch_data)}")

        # Check if the last time value is at a whole second
        if len(self.batch_data) >= self.batch_size and current_time % 1 == 0:
            print("Converting batch to DataFrame")
            self.data_frame_inputs = pd.DataFrame(self.batch_data, columns=['Time', 'wave'] + required_measurements)
            print(self.data_frame_inputs.iloc[0])
            print("data frame shape:", self.data_frame_inputs.shape)
            t_pred, self.y_hat = run_DOLPHINN(self.data_frame_inputs, current_time)
            print("Predicted PtfmTDY:", self.y_hat["PtfmTDY"].iloc[-1])
            print("t_pred:", t_pred.iloc[-1])
            if self.csv_saved is False and current_time == 100:
                self.csv_saved = True
                output_file_path = os.path.join("ZMQ_Prediction_ROSCO", "DOLPHINN", "data", "Control_T100.csv")
                self.data_frame_inputs.to_csv(output_file_path, index=False)
                print("SAVED control CSV at t = 100")

    def setup_zmq_publisher(self):
        print(f"Setting up publisher (Port {self.port})...")
        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        publisher.bind(f"tcp://*:{self.port}")
        return publisher

    def send_delta_B(self, delta_B, topic="delta_B"):
        message = json.dumps({'delta_B': delta_B})
        full_message = f"{topic} {message}"
        self.publisher.send_string(full_message)
        print("SENDING DELTA_B:", delta_B)
        
    def main(self):
        try:
            while True:
                print("t_preds:", self.t_preds)
                print("y_preds:", self.y_preds)
                self.send_delta_B(self.y_hat["PtfmTDY"].iloc[-1], topic="delta_B")
                # Wait some time before checking for new measurements again
                time.sleep(1)  # Adjust sleep time as necessary
        except KeyboardInterrupt:
            print("Terminating publisher...")
            self.publisher.close()

if __name__ == "__main__":
    prediction_instance = PredictionClass()
    prediction_instance.main()