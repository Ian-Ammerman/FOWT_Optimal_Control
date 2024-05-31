import os
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import multiprocessing as mp
from rosco.toolbox.ofTools.case_gen import CaseLibrary as cl
from rosco.toolbox.ofTools.case_gen.run_FAST import run_FAST_ROSCO
from rosco.toolbox.control_interface import wfc_zmq_server
import yaml

class bpcClass:
    def __init__(self, prediction_instance):
        self.prediction_instance = prediction_instance

        # directories
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.rosco_dir = os.path.dirname(self.this_dir)
        self.output_dir = os.path.join(self.this_dir, "Outputs/Driver_DOLPHINN")
        os.makedirs(self.output_dir, exist_ok=True)

        # Specify Load Case (1, 2, 3)
        self.Load_Case = 2

    def run_zmq(self, logfile=None):
        # Start the server at the following address
        network_address = "tcp://*:5555"
        server = wfc_zmq_server(network_address, timeout=1200.0, verbose=False, logfile=logfile)

        # Provide the wind farm control algorithm as the wfc_controller method of the server
        server.wfc_controller = self.wfc_controller

        # Run the server to receive measurements and send setpoints
        server.runserver()

    def wfc_controller(self, id, current_time, measurements):
        # SPECIFY LOAD CASE # IN __INIT__

        # This code is created for BlPitchCMeas Setpoint. However, other DOFs may be selected:
        # FOWT Measurement for Prediction and monitoring - Choose between BlPitchCMeas, PtfmTDX, PtfmTDZ, PtfmTDY, PtfmRDX, PtfmRDY and PtfmRDZ
        FOWT_pred_state = 'BlPitchCMeas'

        # Specify path and load trained DOLPHINN model (Must contain BlPitchCMeas)
        MLSTM_MODEL_NAME = 'TrainingData_Hs_2_75_Tp_6'

        # Specify incoming wave data file name
        # FUTURE_WAVE_FILE = f"WaveData_LC{self.Load_Case}.csv"
        FUTURE_WAVE_FILE = f"WaveData.csv"

        # Retrieve time horizon from trained model
        config_file_path = os.path.join(self.this_dir, "Prediction_Model", "DOLPHINN", "saved_models", f"{MLSTM_MODEL_NAME}", "wave_model", 'config.yaml')
        with open(config_file_path, 'r') as file:
            config_data = yaml.safe_load(file)
        time_horizon = config_data['time_horizon']

        #### Prediction Model Configuration ####
        Prediction = False  # True: Sends prediction offset to ROSCO. False: Deactivate (Pred_Delta_B = 0.0)
        plot_figure = True  # True: Activate real time prediction plotting. False: Deactivate
        Pred_Saturation = False  # True: Saturate prediction offset (Avoid too big angle prediction offset)
        saturation_threshold = 2 * np.pi / 180  # Define the threshold of prediction offset [rad]
        pred_freq = 1  # Defines frequency of calling prediction model
        buffer_duration = time_horizon - 1.0125  # Defines the buffer duration for the prediction before sending offset
        K_pred = 1  # K_pred = 1 for standard calculated offset
        save_csv = True  # Save csv for prediction and measurements
        save_csv_time = 1000  # Specify time for saving csv [s]
        pitch_prediction_error_deg = 3.7

        # Add specified prediction error offset in degrees if coll. blade pitch angle is chosen as FOWT_pred_state
        BlPitchC_Meas_pred_error = pitch_prediction_error_deg if FOWT_pred_state == "BlPitchCMeas" else 0.0

        # Get prediction and predicted time
        Pred_B, t_pred = self.prediction_instance.run_simulation(
            current_time, measurements, plot_figure, time_horizon,
            BlPitchC_Meas_pred_error, pred_freq, save_csv, save_csv_time, FUTURE_WAVE_FILE, FOWT_pred_state, MLSTM_MODEL_NAME)

        # If Blade pitch is predicted: Buffer blade pitch prediction until optimal time to send offset to ROSCO
        if Prediction and FOWT_pred_state == 'BlPitchCMeas': 
            Pred_Delta_B = Buffer(Pred_B, t_pred, current_time, measurements, buffer_duration, BlPitchC_Meas_pred_error, time_horizon) 
            Pred_Delta_B = Saturate(Pred_Delta_B, Pred_Saturation, saturation_threshold)
        else: 
            Pred_Delta_B = 0.0

        # Set control setpoints
        setpoints = {}
        setpoints["ZMQ_YawOffset"] = 0.0
        setpoints['ZMQ_PitOffset(1)'] = Pred_Delta_B * K_pred
        setpoints['ZMQ_PitOffset(2)'] = Pred_Delta_B * K_pred
        setpoints['ZMQ_PitOffset(3)'] = Pred_Delta_B * K_pred

        return setpoints

    def sim_openfast_custom(self):
        print(f"Running custom OpenFAST configuration with Load Case {self.Load_Case}")
        r = run_FAST_ROSCO()
        r.tuning_yaml = "IEA15MW_FOCAL.yaml"
        run_dir = os.path.join(self.output_dir, "Sim_Results")
        r.save_dir = run_dir
        r.wind_case_fcn = cl.custom_wind_wave_case    

        if self.Load_Case == 1:
            wave_height = 1.0
            peak_period = 4.5
        elif self.Load_Case == 2:
            wave_height = 2.0
            peak_period = 5.5
        elif self.Load_Case == 3:
            wave_height = 3.5
            peak_period = 6.5

        r.wind_case_opts = {
            "TMax": 1100,            # Total run time (sec)    
            "wave_height": wave_height,      # WaveHs (meters)       
            "peak_period": peak_period,      # WaveTp (meters)       
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
        
        # Set WaveTMax to TMax + 500
        r.wind_case_opts["WaveTMax"] = r.wind_case_opts["TMax"] + 200
        # Print the updated case options to verify
        
        print("Final case options:", r.wind_case_opts)
        r.controller_params = {"LoggingLevel": 2, "DISCON": {"ZMQ_Mode": 1, "ZMQ_ID": 1, "ZMQ_UpdatePeriod": 0.0125}}
        r.run_FAST()
        
    def sim_openfast(self):
        r = run_FAST_ROSCO()
        r.tuning_yaml = "IEA15MW_FOCAL.yaml"
        run_dir = os.path.join(self.output_dir, "Sim_Results")
        r.save_dir = run_dir
        r.wind_case_fcn = cl.power_curve    

        r.wind_case_opts = {
            "U": [12.5],
            "TMax": 1020,
        }
        r.save_dir = run_dir
        r.controller_params = {}
        r.controller_params["DISCON"] = {}
        r.controller_params["LoggingLevel"] = 2
        r.controller_params["DISCON"]["ZMQ_Mode"] = 1
        r.controller_params["DISCON"]["ZMQ_ID"] = 2
        r.run_FAST()

        
    def main(self):
        logfile = os.path.join(self.output_dir, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    
        # Start the existing processes
        p1 = mp.Process(target=self.run_zmq, args=(logfile,))
        p2 = mp.Process(target=self.sim_openfast_custom)

        p1.start()
        p2.start()
        
        # Wait for multiprocessing processes to complete
        p1.join()
        p2.join()

if __name__ == "__main__":
    from Digital_Twin_ZMQ.Prediction_Model.data_batching import PredictionClass
    from Digital_Twin_ZMQ.Prediction_Model.prediction_functions import Buffer, Saturate

    prediction_instance = PredictionClass()
    bpc = bpcClass(prediction_instance)
    bpc.main()
