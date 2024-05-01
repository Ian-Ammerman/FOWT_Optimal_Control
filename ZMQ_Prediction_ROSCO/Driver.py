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

    def run_zmq(self, logfile=None):
        # Start the server at the following address
        network_address = "tcp://*:5555"
        server = wfc_zmq_server(network_address, timeout=60.0, verbose=False, logfile=logfile)

        # Provide the wind farm control algorithm as the wfc_controller method of the server
        server.wfc_controller = self.wfc_controller

        # Run the server to receive measurements and send setpoints
        server.runserver()

    def wfc_controller(self, id, current_time, measurements):

        delta_B = prediction_instance.run_simulation(current_time, measurements)
            
        if current_time <= 10.0:
            YawOffset = 0.0
        else:
            if id == 1:
                YawOffset = self.DESIRED_YAW_OFFSET[0]
            else:
                YawOffset = self.DESIRED_YAW_OFFSET[1]

        # Use latest_delta_B for blade pitch control
        if delta_B is not None:
            col_pitch_command = delta_B  # Assuming latest_delta_B is already in radians
            # print("Delta_B in Driver.py:", delta_B)
        else: 
            col_pitch_command = 0.0

        # Check if the 'Time' column contains integer values
        if current_time % 10 == 0:
            print("Predicted  Blade Pitch Setpoint:", col_pitch_command)
        
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
