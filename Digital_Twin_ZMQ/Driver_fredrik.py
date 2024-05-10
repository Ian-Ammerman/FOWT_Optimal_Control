# driver_fredrik.py
import os
import zmq
import time
import pandas as pd
import multiprocessing as mp
from ZMQ_Prediction_ROSCO.Fatigue_Estimation.data_monitor import DataMonitor
from ZMQ_Prediction_ROSCO.Fatigue_Estimation.fatigue_damage_RUL_fredrik import RUL_class
from rosco.toolbox.control_interface import wfc_zmq_server 
from rosco.toolbox.ofTools.case_gen import CaseLibrary as cl
from rosco.toolbox.ofTools.case_gen.run_FAST import run_FAST_ROSCO

        
class BladePitchController:
    def __init__(self):
        self.chunk_duration = 3600
        self.rul_instance = RUL_class(emit_callback=self.publish_rul_updates, chunk_duration=self.chunk_duration)
        self.network_address = "tcp://*:5555"
        self.server = None
        
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.this_dir, "Outputs")
        self.input_dir = os.path.join(self.this_dir, "../Outputs")
        self.output_filename = "updated_simulation_data.csv"
        os.makedirs(self.output_dir, exist_ok=True)
        self.logfile = os.path.join(self.output_dir, os.path.splitext(os.path.basename(__file__))[0] + '.log')
        print("Log file path:", self.logfile)

        self.data_monitor = DataMonitor(self.output_dir, self.output_filename, self.chunk_duration)       
        self.data_frame = pd.DataFrame()
        self.last_data_check_time = 10
        
    def run_zmq(self, logfile=None):
        self.context = zmq.Context()
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:5556") 
        
        print("Starting ZMQ server...")
        self.server = wfc_zmq_server(self.network_address, timeout=120.0, verbose=False, logfile=self.logfile)
        self.server.wfc_controller = self.wfc_controller
        self.server.runserver()
            
    def update_system_state(self, current_time):
        # Check if it's time to read new data (every 10 seconds)
        if current_time - self.last_data_check_time >= self.chunk_duration/10:
            new_data = self.data_monitor.read_and_filter_data()
            if not new_data.empty:
                #print(f"Data received for processing at simulation time: {current_time}")
                self.data_frame = pd.concat([self.data_frame, new_data], ignore_index=True)
                self.data_monitor.process_data()  # Process the data if new data was read
                # Pass each row of new_data to the fatigue analysis
                for _, row in new_data.iterrows():
                    self.rul_instance.update_measurements(current_time, row)
                #if current_time >= 50.0:
                #    self.inspect_data()
            else:
                print("No new data available.")
            self.last_data_check_time = current_time
    
    def inspect_data(self):
        if not self.data_frame.empty:
            print("Inspecting data at 50 seconds:")
            print(self.data_frame.head())  # Print the first few rows for a quick check
            csv_path = os.path.join(self.output_dir, "data_at_50_seconds.csv")
            self.data_frame.to_csv(csv_path, index=False)
            print(f"Data saved to {csv_path} for further inspection.")


        
    def publish_rul_updates(self, all_rul_values):
        rul_values_blade_openfast = all_rul_values['rul_values_blade_openfast']
        rul_values_tower_openfast = all_rul_values['rul_values_tower_openfast']
        
        self.pub_socket.send_json({"rul_values_blade_openfast": rul_values_blade_openfast})
        self.pub_socket.send_json({"rul_values_tower_openfast": rul_values_tower_openfast})

    def wfc_controller(self, id, current_time, measurements):
        # Call to process any new data that might have been read
        self.update_system_state(current_time)

        return {'YawOffset': 0.0}
                
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
        r.controller_params["DISCON"]["ZMQ_UpdatePeriod"] = 0.0125
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
