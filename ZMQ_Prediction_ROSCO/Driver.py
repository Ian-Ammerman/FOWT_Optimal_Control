import os
import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import multiprocessing as mp
from rosco.toolbox.ofTools.case_gen import CaseLibrary as cl
from rosco.toolbox.ofTools.case_gen.run_FAST import run_FAST_ROSCO
import zmq
from multiprocessing import Value, Process
from rosco.toolbox.control_interface import wfc_zmq_server
from ZMQ_Prediction_ROSCO.DOLPHINN.prediction.pitch_prediction import PredictionClass
from collections import deque

class bpcClass:
    def __init__(self, prediction_instance):
        self.prediction_instance = prediction_instance
        self.context = zmq.Context()
        self.manager = mp.Manager()
        self.DESIRED_YAW_OFFSET = [-10, 10]
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.rosco_dir = os.path.dirname(self.this_dir)
        self.outputs = os.path.join(self.this_dir, 'bpc_outputs')
        os.makedirs(self.outputs, exist_ok=True)

        # directories
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.rosco_dir = os.path.dirname(self.this_dir)
        self.outputs = os.path.join(self.this_dir, 'bpc_outputs')
        os.makedirs(self.outputs, exist_ok=True)

        # delta_B buffering
        self.delta_B_buffer = deque()
        self.buffer_duration = 20  # Delay duration in seconds
        self.last_used_delta_B = 0.0  # Initially set to zero
        self.last_used_t_pred = None  # Initially set to None
        self.first_delta_received = False
        self.printed_first_delta_B = False
        self.last_whole_second = None  # Track the last whole second for countdown

    def run_zmq(self, logfile=None):
        # Start the server at the following address
        network_address = "tcp://*:5555"
        server = wfc_zmq_server(network_address, timeout=60.0, verbose=False, logfile=logfile)

        # Provide the wind farm control algorithm as the wfc_controller method of the server
        server.wfc_controller = self.wfc_controller

        # Run the server to receive measurements and send setpoints
        server.runserver()

    def wfc_controller(self, id, current_time, measurements):
        # Get prediction and predicted time
        delta_B, t_pred = prediction_instance.run_simulation(current_time, measurements)
        
        # Buffering delta_B with its predicted time and the time it was predicted
        if delta_B is not None:
            self.delta_B_buffer.append((delta_B, current_time + self.buffer_duration, t_pred))
            if not self.first_delta_received and not self.printed_first_delta_B:
                self.printed_first_delta_B = True
                self.first_delta_received = True  # Set the flag on receiving the first delta_B
                self.last_whole_second = int(current_time)  # Initialize countdown start time
                print(f"First delta_B prediction received and buffered: {delta_B} radians at time {current_time}")
        
        # Release buffer based on current time and buffer duration
        while self.delta_B_buffer and self.delta_B_buffer[0][1] <= current_time:
            self.last_used_delta_B, _, self.last_used_t_pred = self.delta_B_buffer.popleft()
            self.first_delta_received = False  # Reset the flag after the first delta_B is used

        # Use the last released delta_B as the control pitch command
        col_pitch_command = self.last_used_delta_B

        # Countdown for the first delta_B in the buffer
        if self.first_delta_received:
            current_whole_second = int(current_time)
            if current_whole_second != self.last_whole_second:
                self.last_whole_second = current_whole_second
                if self.delta_B_buffer:
                    time_to_use = int(self.delta_B_buffer[0][1] - current_time)
                    print(f"Countdown until first delta_B prediction is used: {time_to_use} s")

        # Print the current time and prediction time when a delta_B is used
        if self.last_used_t_pred is not None and current_time % 1 == 0:
            print(f"Current Time: {current_time}, Last Used Prediction Time: {self.last_used_t_pred}")
            print("Sending Predicted Blade Pitch Setpoint:", col_pitch_command)
        elif self.last_used_t_pred is None and current_time % 5 == 0:
            print("Blade Pitch Setpoint:", col_pitch_command)

        # Yaw offset handling based on id
        YawOffset = self.DESIRED_YAW_OFFSET[0] if id == 1 else self.DESIRED_YAW_OFFSET[1] if current_time > 10.0 else 0.0

        # Set control setpoints
        setpoints = {}
        setpoints["ZMQ_YawOffset"] = YawOffset
        setpoints['ZMQ_PitOffset(1)'] = col_pitch_command
        setpoints['ZMQ_PitOffset(2)'] = col_pitch_command
        setpoints['ZMQ_PitOffset(3)'] = col_pitch_command

        return setpoints

    
    def sim_openfast(self):
        # Create an instance of the FAST simulation with ROSCO controller
        r = run_FAST_ROSCO()
        r.tuning_yaml = "IEA15MW_FOCAL.yaml"
        r.wind_case_fcn = cl.power_curve
        r.wind_case_opts = {
            "U": [8],
            "TMax": 1000,
        }
        run_dir = os.path.join(self.outputs, "FOCAL")
        r.controller_params = {}
        r.controller_params["LoggingLevel"] = 2
        r.controller_params["DISCON"] = {}
        r.controller_params["DISCON"]["ZMQ_Mode"] = 1
        r.controller_params["DISCON"]["ZMQ_ID"] = 1
        r.controller_params["DISCON"]["ZMQ_UpdatePeriod"] = 0.0125
        r.save_dir = run_dir
        r.run_FAST()

    def main(self):
        logfile = os.path.join(self.outputs, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    
        print("Started pitch_prediction.py subprocess.")

        # Start the existing processes
        p1 = mp.Process(target=self.run_zmq, args=(logfile,))
        p2 = mp.Process(target=self.sim_openfast)

        p1.start()
        p2.start()
        
        # Wait for multiprocessing processes to complete
        p1.join()
        p2.join()

if __name__ == "__main__":
    prediction_instance = PredictionClass()
    bpc = bpcClass(prediction_instance)
    bpc.main()
