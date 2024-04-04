import zmq
import json
import sys
import time
import os
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append('/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO')
from MLSTM_WRP.MLSTM_04_FC4_RT import run_MLSTM
print("RUNNING PITCH_PREDICTION.PY")

import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class PredictionClass():
    def __init__(self):
        self.port = "5556"
        self.publisher = self.setup_zmq_publisher()
        self.batch_data = []  # To store incoming data temporarily
        self.batch_size = 10  # Number of rows or timesteps per batch
        self.file_generation_count = 0  # Initialize a counter for file generation
        self.temp_csv_files = []  # List to keep track of the temporary CSV file paths


    def run_simulation(self, current_time, measurements):
        print("RUN_SIMULATION CALLED")

        temp_csv_dir = '/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO/MLSTM_WRP/Data/Batch_Dir'
        os.makedirs(temp_csv_dir, exist_ok=True)
        
        # Load the CSV file once to optimize performance
        if not hasattr(self, 'csv_df'):
            csv_file_path = '/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO/MLSTM_WRP/Data/FC4/windwave_test.csv'
            self.csv_df = pd.read_csv(csv_file_path)
        
        y, yhat = None, None

        required_measurements = [
            'PtfmTDX', 'PtfmTDZ', 'PtfmRDY', 'GenTqMeas', 'RotSpeed', 'NacIMU_FA_Acc', 'PtfmRDY', 'PtfmRDZ'
        ]

        # Real-time measurements
        measurement_values = [measurements.get(key, 0.0) for key in required_measurements]

        # Determine the closest whole second for the current iteration
        # This assumes 'current_time' is in seconds and increases by a fixed step between calls
        target_second = round(current_time)
        
        # Find the row in the CSV where 'Time' is closest to this whole second
        closest_row = self.csv_df.iloc[(self.csv_df['Time'] - target_second).abs().argmin()]
        waveStaff5_measurement = closest_row['waveStaff5']

        # Append current timestep data, including "waveStaff5"
        self.batch_data.append([current_time] + measurement_values + [waveStaff5_measurement])

        if len(self.batch_data) >= self.batch_size:
                # Convert batch_data to DataFrame, adding "waveStaff5" to the columns list
                data_frame_inputs = pd.DataFrame(self.batch_data, columns=['Time'] + required_measurements + ['waveStaff5'])

                # Save this batch as a temporary CSV file
                temp_csv_path = os.path.join(temp_csv_dir, f'batch_data_{current_time}.csv')
                data_frame_inputs.to_csv(temp_csv_path, index=False)
                self.temp_csv_files.append(temp_csv_path)  # Keep track of the generated file

                try:
                    MODEL_PATH = "/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO/MLSTM_WRP/models/FC4_OPT_MLSTM_WRP_8dof_T20_nm_0.39_dt_1.0_wvshftn_0.75"
                    SCALER_PATH = os.path.join(MODEL_PATH, "scalers", "scaler.pkl")
                    
                    # Now, instead of directly passing the DataFrame, you pass the path to the CSV file
                    # This assumes run_MLSTM can work with CSV file paths or you adapt it to read the CSV within its logic
                    y, yhat = run_MLSTM(1700, 2000, 1.0, 4, MODEL_PATH, SCALER_PATH, temp_csv_path, 20)
                    
                    print("Simulation completed.")
                    print("yhat:", yhat)
                except Exception as e:
                    print(f"Error during simulation: {e}")
                    y, yhat = None, None

                # Increment the file generation counter
                self.file_generation_count += 1

                # Delete the oldest file every 6th new file
                if self.file_generation_count % 2 == 0 and self.temp_csv_files:
                    oldest_file = self.temp_csv_files.pop(0)  # Remove the oldest file from the tracking list
                    if os.path.exists(oldest_file):
                        os.remove(oldest_file)  # Delete the oldest file
                
                # Reset batch_data for the next batch
                self.batch_data = []
                

                return y, yhat


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

    def main(self, y, yhat):
        try:
            while True:
                print("y:", y)
                print("yhat:", yhat)
                self.send_delta_B(yhat[1], topic="delta_B")
                # Wait some time before checking for new measurements again
                time.sleep(1)  # Adjust sleep time as necessary
        except KeyboardInterrupt:
            print("Terminating publisher...")
            self.publisher.close()

if __name__ == "__main__":
    prediction_instance = predictionClass()
    prediction_instance.main()
