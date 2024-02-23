import zmq
import json  # Assuming data is sent as JSON strings
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import logging
import time
from MLSTM_WRP.MLSTM_04_FC4_RT import run_MLSTM
print("RUNNING PITCH_PREDICTION.PY")

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


class predictionClass():

    def __init__(self):
        self.port = "5556"
        self.publisher = self.setup_zmq_publisher()

    def update_measurements(self, current_time, measurements):

        # Debugging: Print or log the received measurements for verification
        # print("current time:", current_time)
        print({measurements['PtfmTDX']})
        pass


    TEST_NUM = 4
    MODEL_PATH = os.path.join("MLSTM_WRP", "models", "FC4_OPT_MLSTM_WRP_8dof_T20_nm_0.39_dt_1.0_wvshftn_0.75")
    SCALER_PATH = os.path.join("MLSTM_WRP", "models", "FC4_OPT_MLSTM_WRP_8dof_T20_nm_0.39_dt_1.0_wvshftn_0.75", "scalers", "scaler.pkl")
    DUMMY_DATA_INPUT_FILE = os.path.join("MLSTM_WRP", "Data", "FC4", "windwave_test.csv")
    TIME_HORIZON = 20
    start_simulation = 1700
    end_simulation = 2000
    timestep = 1.0

    y, yhat = run_MLSTM(start_simulation, end_simulation, timestep, TEST_NUM, MODEL_PATH, SCALER_PATH, DUMMY_DATA_INPUT_FILE, TIME_HORIZON)

    
    def setup_zmq_publisher(self):
        print(f"Setting up publisher (Port {self.port})...")
        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        publisher.bind(f"tcp://*:{self.port}")
        return publisher


    def send_delta_B(self, delta_B, topic="delta_B"):
        message = json.dumps({'delta_B': delta_B})
        full_message = f"{topic} {message}"# Combine topic and message
        self.publisher.send_string(full_message)
        # print(full_message)

    def main(self):
        delta_B_counter = 1  # Initialize the counter at 1
        try:
            while True:

                self.send_delta_B(delta_B_counter, topic="delta_B")
                # print(f"Publishing delta_B: {delta_B_counter}")  # Optional: print the published value
                delta_B_counter += 1  # Increment the counter
                time.sleep(1)  # Add a delay to simulate time between predictions, adjust as needed
        except KeyboardInterrupt:
            print("Terminating publisher...")
            self.publisher.close()

if __name__ == "__main__":
    prediction_instance = predictionClass()
    prediction_instance.main()
