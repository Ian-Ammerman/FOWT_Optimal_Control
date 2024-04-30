import zmq
import json
import sys
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from ZMQ_Prediction_ROSCO.DOLPHINN.vmod.dolphinn import DOLPHINN as DOL
from ZMQ_Prediction_ROSCO.DOLPHINN.examples.MLSTM import run_prediction
print("RUNNING PITCH_PREDICTION.PY?")

# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class PredictionClass():
    def __init__(self):
        self.port = "5556"
        self.publisher = self.setup_zmq_publisher()
        self.batch_data = []  # To store incoming data temporarily
        self.batch_size = 100  # Number of rows or timesteps per batch
        self.file_generation_count = 0  # Initialize a counter for file generation
        self.t_preds = None
        self.y_preds = None
        self.has_run_once = False  # Flag to check if simulation has run once
        self.csv_saved = False  # Flag to check if the CSV has been saved once

    def run_simulation(self, current_time, measurements):
        
        # Load the CSV file once to optimize performance
        if not hasattr(self, 'csv_df'):
            csv_file_path = '/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO/DOLPHINN/data/wave_data_00125_500k.csv'
            self.csv_df = pd.read_csv(csv_file_path)
        
        required_measurements = [
            'PtfmTDX', 'PtfmTDY', 'PtfmTDZ', 'PtfmRDX', 'PtfmRDY', 'PtfmRDZ', 'FA_Acc'
        ]

        # Real-time measurements
        measurement_values = [measurements.get(key, 0.0) for key in required_measurements]

        # Find the row in the CSV where 'Time' is closest to a whole second
        # Append current timestep data, including "wave"
        wave_measurement = self.csv_df.loc[self.csv_df['Time'] == current_time, 'wave'].iloc[0]

        self.batch_data.append([current_time] + measurement_values + [wave_measurement])

        if len(self.batch_data) > self.batch_size:
        # Remove the oldest row when exceeding batch_size
            self.batch_data.pop(0)

        # Convert batch_data to DataFrame, adding "wave" to the columns list
        data_frame_inputs = pd.DataFrame(self.batch_data, columns=['Time'] + required_measurements + ['wave'])

        #if len(self.batch_data) >= self.batch_size and not self.csv_saved:
         #   self.csv_saved = True  # Update the flag to True to prevent future saves
          #  output_file_path = '/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO/DOLPHINN/data/data_frame_inputs.csv'
           # data_frame_inputs.to_csv(output_file_path, index=False)

        # Get the first and last time values
        oldest_time = data_frame_inputs["Time"].iloc[0]
        newest_time = data_frame_inputs["Time"].iloc[-1]

        # Check if both times are whole seconds and print them if they are            
        if newest_time % 1 == 0:
            print("Newest row, Time:", newest_time)
            print("Oldest row, Time:", oldest_time)
            print("Data frame shape:", data_frame_inputs.shape)

        if len(data_frame_inputs) >= self.batch_size and not self.has_run_once:
            self.has_run_once = True

            run_prediction()
            


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
                run_prediction()
                print("t_preds:", self.t_preds)
                print("y_preds:", self.y_preds)
                # self.send_delta_B(self.y_preds[1], topic="delta_B")
                # Wait some time before checking for new measurements again
                time.sleep(1)  # Adjust sleep time as necessary
        except KeyboardInterrupt:
            print("Terminating publisher...")
            self.publisher.close()

if __name__ == "__main__":
    prediction_instance = PredictionClass()
    prediction_instance.main()