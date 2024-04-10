# ffs_blade_pitch_controller.py
import eventlet
eventlet.monkey_patch()

import os
import zmq
import time
import numpy as np
import multiprocessing as mp
from fatigue_damage_RUL import RUL_class
from real_time_server import RealTimeServer_class
from rosco.toolbox.control_interface import wfc_zmq_server 
from rosco.toolbox.ofTools.case_gen import CaseLibrary as cl
from rosco.toolbox.ofTools.case_gen.run_FAST import run_FAST_ROSCO

DESIRED_YAW_OFFSET = [-10, 10]


class BladePitchController:
    def __init__(self):
        print("Setting up RUL_class with publish_rul_updates callback.")
        self.rul_instance = RUL_class(emit_callback=self.publish_rul_updates)
        self.network_address = "tcp://*:5555"
        self.server = None
        
    def run_zmq(self, logfile=None):
       # ZeroMQ context and socket for publishing RUL updates
        self.context = zmq.Context()
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:5556")  # Publish RUL updates on port 5556
        
        print("Starting ZMQ server...")
        self.server = wfc_zmq_server(self.network_address, timeout=120.0, verbose=False, logfile=logfile)
        self.server.wfc_controller = self.wfc_controller
        self.server.runserver()
            
    def publish_rul_updates(self, rul_values):
        # Publish the RUL values using ZeroMQ
        #print(f"publish_rul_updates called with rul_values: {rul_values}")
        self.pub_socket.send_json({"rul_values": rul_values})
        
    def wfc_controller(self, id, current_time, measurements):
        self.rul_instance.update_measurements(current_time, measurements)
        return {'YawOffset': 0.0}
                
    def sim_openfast(self, this_dir):
        r = run_FAST_ROSCO()
        r.tuning_yaml = "IEA15MW_FOCAL.yaml"
        r.run_dir = os.path.join(this_dir, "SimulationOutputs")
        r.wind_case_fcn = cl.power_curve
        r.wind_case_opts = {"TMax": 7210}
        r.controller_params = {
            "LoggingLevel": 2,
            "DISCON": {"ZMQ_Mode": 1, "ZMQ_ID": 1}
        }
        r.run_FAST()
        
    def main(self):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        outputs = os.path.join(this_dir, 'examples_out')
        os.makedirs(outputs, exist_ok=True)

        processes = [
            mp.Process(target=self.run_zmq, args=(os.path.join(outputs, 'zmq.log'),)),
            mp.Process(target=self.sim_openfast, args=(outputs,)), 
            mp.Process(target=self.publish_rul_updates)
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join()



if __name__ == '__main__':
    controller = BladePitchController()
    controller.main()
