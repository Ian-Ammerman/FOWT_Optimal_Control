import zmq
import json
import sys
import time
import os
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append('/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO')
from MLSTM_WRP.MLSTM_04_FC4_RT import run_MLSTM
print("RUNNING PITCH_PREDICTION.PY")

import logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class predictionClass():
    def __init__(self):
        self.port = "5556"
        self.publisher = self.setup_zmq_publisher()
#        self.DUMMY_DATA_INPUTS = {}

    def run_simulation(self, current_time, measurements):
        print("RUN_SIMULATION CALLED")

        required_measurements = [
            'PtfmTDX', 'PtfmTDZ', 'PtfmRDY', 'GenTqMeas', 'RotSpeed', 'NacIMU_FA_Acc', 'PtfmRDY'
        ]

        measurement_values = [measurements.get(key, 0.0) for key in required_measurements]

        current_time_list = [current_time]  

        data_frame_inputs = pd.DataFrame([current_time_list + measurement_values], columns=['Time'] + required_measurements)

        print("Starting simulation...")

        MODEL_PATH = "/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO/MLSTM_WRP/models/FC4_OPT_MLSTM_WRP_8dof_T20_nm_0.39_dt_1.0_wvshftn_0.75"
        SCALER_PATH = os.path.join(MODEL_PATH, "scalers", "scaler.pkl")
        try:
            print("Measurements updated, READY for simulation:", data_frame_inputs.values.tolist())

            y, yhat = run_MLSTM(1700, 2000, 1.0, 4, MODEL_PATH, SCALER_PATH, data_frame_inputs, 20)
            print("Simulation completed.")
            print("yhat:", yhat)
        except Exception as e:
            print(f"Error during simulation: {e}")
            y, yhat = None, None  # You might choose to handle this differently depending on your error handling strategy

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
