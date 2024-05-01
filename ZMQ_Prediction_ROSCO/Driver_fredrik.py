# driver_fredrik.py
import os
import zmq
import pandas as pd
import multiprocessing as mp
from ZMQ_Prediction_ROSCO.FredrikPart.fatigue_damage_RUL_fredrik import RUL_class
from rosco.toolbox.control_interface import wfc_zmq_server 
from rosco.toolbox.ofTools.case_gen import CaseLibrary as cl
from rosco.toolbox.ofTools.case_gen.run_FAST import run_FAST_ROSCO

        
class BladePitchController:
    def __init__(self):
        self.rul_instance = RUL_class(emit_callback=self.publish_rul_updates)
        self.network_address = "tcp://*:5555"
        self.server = None
        
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.this_dir, "Outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.logfile = os.path.join(self.output_dir, os.path.splitext(os.path.basename(__file__))[0] + '.log')
        print("Log file path:", self.logfile)
        
        self.data = pd.read_csv('ZMQ_Prediction_ROSCO/FredrikPart/csvfiles/Results24hour_moments.csv')
        self.last_processed_index = 0  
        
    def run_zmq(self, logfile=None):
        self.context = zmq.Context()
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:5556") 
        
        print("Starting ZMQ server...")
        self.server = wfc_zmq_server(self.network_address, timeout=120.0, verbose=False, logfile=self.logfile)
        self.server.wfc_controller = self.wfc_controller
        self.server.runserver()
            
    def simulate_real_time_data_stream(self, current_sim_time):
        new_data = self.data.iloc[self.last_processed_index:]
        new_data = new_data[new_data['Time'] <= current_sim_time]
        self.last_processed_index += len(new_data)
        
        for _, row in new_data.iterrows():
            csv_measurements = {
                'Time': row['Time'],
                'RootFzb1': row['RootFzb1'],  # Axial shear force, blade 1
                'RootMxb1': row['RootMxb1'],  # Local bending moment x (edgewise), blade 1
                'RootMyb1': row['RootMyb1'],  # Local bending moment y (flapwise), blade 1
                'RootFzb2': row['RootFzb2'],  # Axial shear force, blade 2
                'RootMxb2': row['RootMxb2'],  # Local bending moment x (edgewise), blade 2
                'RootMyb2': row['RootMyb2'],  # Local bending moment y (flapwise), blade 2
                'RootFzb3': row['RootFzb3'],  # Axial shear force, blade 3
                'RootMxb3': row['RootMxb3'],  # Local bending moment x (edgewise), blade 3
                'RootMyb3': row['RootMyb3'],  # Local bending moment y (flapwise), blade 3
                'TwrBsFzt': row['TwrBsFzt'],  # Axial shear force, tower base
                'TwrBsMxt': row['TwrBsMxt'],  # Local bending moment x (edgewise), tower base
                'TwrBsMyt': row['TwrBsMyt'],  # Local bending moment y (flapwise), tower base
            }
            self.rul_instance.update_measurements_from_csv(current_sim_time, csv_measurements)

    def publish_rul_updates(self, all_rul_values):
        # Debug print, uncomment to check incoming data structure
        # print(f"publish_rul_updates called with RUL values: {all_rul_values}")
        
        # Extract each set of RUL values from the dictionary
        # rul_values_blade_rosco = all_rul_values['rul_values_blade_rosco']
        rul_values_blade_openfast = all_rul_values['rul_values_blade_openfast']
        rul_values_tower_openfast = all_rul_values['rul_values_tower_openfast']
        
        # Publish each set of RUL values as a separate JSON message
        # self.pub_socket.send_json({"rul_values_blade_rosco": rul_values_blade_rosco})
        self.pub_socket.send_json({"rul_values_blade_openfast": rul_values_blade_openfast})
        self.pub_socket.send_json({"rul_values_tower_openfast": rul_values_tower_openfast})

    def wfc_controller(self, id, current_time, measurements):
        #self.rul_instance.update_measurements(current_time, measurements)
        self.simulate_real_time_data_stream(current_time)  
    
        return {'YawOffset': 0.0}  # This is kept to keep wfc going. YawOffset not needed
                
    def sim_openfast(self):
        r = run_FAST_ROSCO()
        r.tuning_yaml = "IEA15MW_FOCAL.yaml"
        run_dir = os.path.join(self.output_dir, "Sim_Results")
        r.save_dir = run_dir
        r.wind_case_fcn = cl.power_curve
        r.wind_case_opts = {"TMax": 86400}
        r.controller_params = {
            "LoggingLevel": 2,
            "DISCON": {"ZMQ_Mode": 1, "ZMQ_ID": 1}
        }
        r.run_FAST()
        
    def main(self):    
        processes = [
            mp.Process(target=self.run_zmq),
            mp.Process(target=self.sim_openfast) 
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

if __name__ == '__main__':
    controller = BladePitchController()
    controller.main()
