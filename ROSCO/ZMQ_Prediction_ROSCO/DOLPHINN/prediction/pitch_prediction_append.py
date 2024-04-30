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
        self.batch_data = []  # To store incoming data temporarily
        self.batch_size = 6000  # Number of rows or timesteps per batch
        self.file_generation_count = 0  # Initialize a counter for file generation
        self.t_preds = None
        self.y_preds = None
        self.has_run_once = False  # Flag to check if simulation has run once
        self.csv_saved = False  # Flag to check if the CSV has been saved once
        self.initial_time = None
        self.time_horizon = 20

    def run_simulation(self, current_time, measurements):
        # Load the CSV file once to optimize performance
        #if not hasattr(self, 'csv_df'):
            #print("Retrieving wave data ...")
            #csv_file_path = os.path.join("ZMQ_Prediction_ROSCO", "DOLPHINN", "data", "wave_data_00125_500k.csv")
            #self.csv_df = pd.read_csv(csv_file_path)

        required_measurements = [
            'PtfmTDX', 'PtfmTDY', 'PtfmTDZ', 'PtfmRDX', 'PtfmRDY', 'PtfmRDZ', 'FA_Acc'
        ]

        # Set initial time if it has not been set
        if self.initial_time is None:
            self.initial_time = current_time

        elapsed_time = current_time - self.initial_time

        # Real-time measurements
        measurement_values = [measurements.get(key, 0.0) for key in required_measurements]

        # wave_measurement = self.csv_df.loc[self.csv_df['Time'] == current_time, 'wave'].iloc[0]
        wave_measurement = self.get_wave_measurement(current_time)

        # For all timesteps, append Time and wave
        self.batch_data.append([current_time, wave_measurement] + [None] * len(required_measurements))

        # After 20 seconds, start updating measurements from 20 seconds ago
        if elapsed_time >= self.time_horizon:
            # The data now should be updated to include actual measurements at their respective delayed times
            for i, row in enumerate(self.batch_data):
                row_time = row[0]
                if row_time <= current_time - self.time_horizon:
                    # Find the corresponding measurements that are 20 seconds older
                    index_of_measurements = next((index for index, time_measure in enumerate(self.batch_data) if time_measure[0] == row_time), None)
                    if index_of_measurements is not None:
                        self.batch_data[index_of_measurements] = [self.batch_data[index_of_measurements][0], self.batch_data[index_of_measurements][1]] + measurement_values

        if len(self.batch_data) > self.batch_size:
            # Remove the oldest row when exceeding batch_size
            self.batch_data.pop(0)

        # Ensure columns are set correctly based on the data condition
        if elapsed_time >= self.time_horizon:
            columns = ['Time', 'wave'] + required_measurements
        else:
            columns = ['Time', 'wave'] + ['None'] * len(required_measurements)

        # Convert batch_data to DataFrame, adding "wave" to the columns list
        data_frame_inputs = pd.DataFrame(self.batch_data, columns=columns)
        # print(data_frame_inputs)

        if len(self.batch_data) >= self.batch_size and not self.csv_saved:
            self.csv_saved = True  # Update the flag to True to prevent future saves
            output_file_path = '/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO/DOLPHINN/data/test_csv_NAN.csv'
            data_frame_inputs.to_csv(output_file_path, index=False)

        # Get the first and last time values
        oldest_time = data_frame_inputs["Time"].iloc[0]
        newest_time = data_frame_inputs["Time"].iloc[-1]

        # Check if both times are whole seconds and print them if they are            
        if newest_time % 1 == 0:
            print("Newest row, Time:", newest_time)
            print("Oldest row, Time:", oldest_time)
            print("Data frame shape:", data_frame_inputs.shape)

        if len(data_frame_inputs) >= self.batch_size:
            # Get the last time value from the DataFrame
            last_time = data_frame_inputs["Time"].iloc[-1]
            # Check if the last time value is at a whole second
            if last_time % 1 == 0:
                t_pred, self.y_hat = run_DOLPHINN(data_frame_inputs, current_time)
                print("Predicted PtfmTDY:", self.y_hat["PtfmTDY"].iloc[-1])
                print("t_pred:", t_pred.iloc[-1])

    def get_wave_measurement(self, current_time):
        wave = np.sin(current_time)
        return wave

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