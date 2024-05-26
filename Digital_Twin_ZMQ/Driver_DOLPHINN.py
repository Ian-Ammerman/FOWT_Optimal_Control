import os
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import multiprocessing as mp
from rosco.toolbox.ofTools.case_gen import CaseLibrary as cl
from rosco.toolbox.ofTools.case_gen.run_FAST import run_FAST_ROSCO
from rosco.toolbox.control_interface import wfc_zmq_server
from Digital_Twin_ZMQ.Blade_Pitch_Prediction.pitch_prediction import PredictionClass
from Digital_Twin_ZMQ.Blade_Pitch_Prediction.prediction_functions import Buffer, Saturate

import yaml

class bpcClass:
    def __init__(self, prediction_instance):
        self.prediction_instance = prediction_instance

        # directories
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.rosco_dir = os.path.dirname(self.this_dir)
        self.output_dir = os.path.join('Sim_output_dir')
        os.makedirs(self.output_dir, exist_ok=True)

    def run_zmq(self, logfile=None):
        # Start the server at the following address
        network_address = "tcp://*:5555"
        server = wfc_zmq_server(network_address, timeout=1200.0, verbose=False, logfile=logfile)

        # Provide the wind farm control algorithm as the wfc_controller method of the server
        server.wfc_controller = self.wfc_controller

        # Run the server to receive measurements and send setpoints
        server.runserver()

    def wfc_controller(self, id, current_time, measurements):
        # Specify path and load trained DOLPHINN model (Must contain BlPitchCMeas)
        DOLPHINN_PATH = os.path.join("Digital_Twin_ZMQ", "Blade_Pitch_Prediction", "DOLPHINN", "saved_models", "TrainingData_Hs_2_75_Tp_6", "wave_model")
        config_file_path = os.path.join(DOLPHINN_PATH, 'config.yaml')
        with open(config_file_path, 'r') as file:
            config_data = yaml.safe_load(file)
        time_horizon = config_data['time_horizon']

        #### Prediction Model Configuration ####
        Prediction = False # True: Sends prediction offset to ROSCO. False: Deactivate (Pred_Delta_B = 0.0)
        plot_figure = False # True: Activate real time prction plotting. False: Deactivate
        Pred_Saturation = False # True: Saturate prediction offset (Avoid too big angle prediction offset)
        saturation_treshold = 2*np.pi/180 # Define the treshold of prediction offset [rad]
        pred_error = 1.4 # Defines the offset for the trained model, found from training_results [deg]
        pred_freq = 1 # Defines frequency of calling prediction model
        buffer_duration = time_horizon - 1.0125 # Defines the buffer duration for the prediction before sending offset
        weighting = True

        # Save measurements and prediction as csv
        save_csv = True
        save_csv_time = 1000

        # Get prediction and predicted time
        Pred_B, t_pred = self.prediction_instance.run_simulation(current_time, measurements, DOLPHINN_PATH, plot_figure, time_horizon, pred_error, pred_freq, save_csv, save_csv_time)
        # Buffer prediction until optimal time to send offset to ROSCO
        if Prediction: 
            Pred_Delta_B = Buffer(Pred_B, t_pred, current_time, measurements, buffer_duration, pred_error, time_horizon) 
            Pred_Delta_B = Saturate(Pred_Delta_B, Pred_Saturation, saturation_treshold)
        else: 
            Pred_Delta_B = 0.0
        
        if weighting:
            W = 100
        else:
            W = 1

        # Set control setpoints
        setpoints = {}
        setpoints["ZMQ_YawOffset"] = 0.0
        setpoints['ZMQ_PitOffset(1)'] = Pred_Delta_B * W
        setpoints['ZMQ_PitOffset(2)'] = Pred_Delta_B * W
        setpoints['ZMQ_PitOffset(3)'] = Pred_Delta_B * W

        return setpoints

    def sim_openfast(self):
        r = run_FAST_ROSCO()
        r.tuning_yaml = "IEA15MW_FOCAL.yaml"
        run_dir = os.path.join(self.output_dir, "Sim_Results")
        r.save_dir = run_dir
        r.wind_case_fcn = cl.custom_wind_wave_case    
        
        r.wind_case_opts = {
            "TMax": 1020,            # Total run time (sec)    
            "wave_height": 1.0,      # WaveHs (meters)       
            "peak_period": 4.5,      # WaveTp (meters)       
            "wave_direction": 0,       # WaveDir (degrees)  
            "WvDiffQTF": "False",       # 2nd order wave diffraction term
            "WvSumQTF": "False"         # 2nd order wave sum-frequency term
        }  
        self.steady_wind = True
        if self.steady_wind:
            print("Setting options for steady wind")
            r.wind_case_opts.update({"wind_type": 1, "HWindSpeed": 12.5})  # Horizontal windspeed (m/s)
        else:
            print("Setting options for turbulent wind")
            r.wind_case_opts.update({"turb_wind_speed": "20" })             # TurbSim Full-Field   (m/s) 

        # Print the updated case options to verify
        print("Final case options:", r.wind_case_opts)
        r.controller_params = {"LoggingLevel": 2, "DISCON": {"ZMQ_Mode": 1, "ZMQ_ID": 1, "ZMQ_UpdatePeriod": 0.0125}}
        r.run_FAST()
        
    def main(self):
        logfile = os.path.join(self.output_dir, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    
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

    
