import os
import sys
import numpy as np
import zmq
import pandas as pd
import multiprocessing as mp
import yaml
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from rosco.toolbox.ofTools.case_gen import CaseLibrary as cl
from rosco.toolbox.ofTools.case_gen.run_FAST import run_FAST_ROSCO
from rosco.toolbox.control_interface import wfc_zmq_server
from Prediction_Model.data_batching import PredictionClass
from Prediction_Model.prediction_functions import Buffer, Saturate
from Fatigue_Estimation_Model.data_collecting import DataMonitor
from Fatigue_Estimation_Model.fatigue_damage_RUL import RUL_class
from Live_Monitoring.real_time_server import RealTimeServer_class


class CombinedController:
    def __init__(self):
        
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.rosco_dir = os.path.dirname(self.this_dir)
        self.output_dir = os.path.join(self.this_dir, "Outputs/Driver_Fatigue")
        self.input_dir = os.path.join(self.this_dir, "../Outputs/Driver_Fatigue")
        os.makedirs(self.output_dir, exist_ok=True)
        self.Load_Case = 2

        self.network_address = "tcp://*:5555"
        self.chunk_duration = 20
        self.nominal_design_life = 20
        self.logfile = os.path.join(self.output_dir, os.path.splitext(os.path.basename(__file__))[0] + '.log')
        self.csv_file_path = os.path.join(self.output_dir, 'rul_values.csv')
        
        self.prediction_instance = PredictionClass()
        self.rul_instance = RUL_class(emit_callback=self.publish_rul_updates, chunk_duration=self.chunk_duration, nominal_design_life_years=self.nominal_design_life)
        self.output_filename = "updated_simulation_data.csv"

        self.data_collecting = DataMonitor(self.output_dir, self.output_filename, self.chunk_duration)       
        self.data_frame = pd.DataFrame()
        self.last_data_check_time = 10

    def run_zmq(self, logfile=None):
        self.context = zmq.Context()
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://*:5556") 
        
        print("Starting ZMQ server...")
        self.server = wfc_zmq_server(self.network_address, timeout=1200.0, verbose=False, logfile=self.logfile)
        self.server.wfc_controller = self.wfc_controller
        self.server.runserver()

    def wfc_controller(self, id, current_time, measurements):
        self.update_system_state(current_time, save_to_csv=True)
        setpoints = self.prediction_setpoints(current_time, measurements)
        return setpoints

    def prediction_setpoints(self, current_time, measurements):
        FOWT_pred_state = 'BlPitchCMeas'
        MLSTM_MODEL_NAME = 'TrainingData_Hs_2_75_Tp_6'
        FUTURE_WAVE_FILE = f"WaveData.csv"

        config_file_path = os.path.join(self.this_dir, "Prediction_Model", "DOLPHINN", "saved_models", f"{MLSTM_MODEL_NAME}", "wave_model", 'config.yaml')
        with open(config_file_path, 'r') as file:
            config_data = yaml.safe_load(file)
        time_horizon = config_data['time_horizon']

        Prediction = False
        plot_figure = False
        Pred_Saturation = False
        saturation_threshold = 2 * np.pi / 180
        pred_freq = 1
        buffer_duration = time_horizon - 1.0125
        K_pred = 1
        save_csv = True
        save_csv_time = 1000
        pitch_prediction_error_deg = 3.7
        BlPitchCMeas_pred_error = pitch_prediction_error_deg if FOWT_pred_state == "BlPitchCMeas" else 0.0

        y_hat_raw, t_pred_raw = self.prediction_instance.run_simulation(
            current_time, measurements, plot_figure, time_horizon,
            BlPitchCMeas_pred_error, pred_freq, save_csv, save_csv_time, 
            FUTURE_WAVE_FILE, FOWT_pred_state, MLSTM_MODEL_NAME)

        if y_hat_raw is not None: 
            Pred_B = y_hat_raw[f"{FOWT_pred_state}"].iloc[-1]
            t_pred = t_pred_raw.iloc[-1]
            present_state_web = measurements[f"{FOWT_pred_state}"]
        else:
            Pred_B = None
            t_pred = None
            present_state_web = None

        if present_state_web is not None:
            if FOWT_pred_state in ["BlPitchCMeas", "PtfmRDX", "PtfmRDY", "PtfmRDZ"]:
                present_state_web = present_state_web * 180 / np.pi


        if FOWT_pred_state == 'BlPitchCMeas':
            Pred_Delta_B, Pred_B_Buffered = Buffer(Pred_B, t_pred, current_time, measurements, buffer_duration, BlPitchCMeas_pred_error, time_horizon, Prediction)
            Pred_Delta_B = Saturate(Pred_Delta_B, Pred_Saturation, saturation_threshold)
            if Prediction:
                Pred_Delta_B_setpoint = Pred_Delta_B
            else:
                Pred_Delta_B_setpoint = 0.0
        else:
            Pred_Delta_B_Setpoint = 0.0
        
        RotSpeed = measurements["RotSpeed"]
        WE_Vw = measurements["WE_Vw"]
        VS_GenPwr = measurements["VS_GenPwr"]
        if y_hat_raw is not None and FOWT_pred_state == 'BlPitchCMeas':
            self.publish_prediction(Pred_B + BlPitchCMeas_pred_error, t_pred, current_time, present_state_web, time_horizon, Pred_Delta_B, Pred_B_Buffered, RotSpeed, WE_Vw, VS_GenPwr)
        else:
            self.publish_prediction(Pred_B, t_pred, current_time, present_state_web, time_horizon, Pred_Delta_B, Pred_B_Buffered, RotSpeed, WE_Vw, VS_GenPwr)

        setpoints = {
            "ZMQ_YawOffset": 0.0,
            'ZMQ_PitOffset(1)': Pred_Delta_B_setpoint * K_pred,
            'ZMQ_PitOffset(2)': Pred_Delta_B_setpoint * K_pred,
            'ZMQ_PitOffset(3)': Pred_Delta_B_setpoint * K_pred
        }
        return setpoints

    def update_system_state(self, current_time, save_to_csv=False, csv_file_path=None):
        if current_time - self.last_data_check_time >= self.chunk_duration/10:
            new_data = self.data_collecting.read_and_filter_data()
            if not new_data.empty: 
                print(f"Data received for processing at simulation time: {current_time}")
                self.data_frame = pd.concat([self.data_frame, new_data], ignore_index=True)
                self.data_collecting.process_data()  # Process the data if new data was read
                # Pass each row of new_data to the fatigue analysis
                for _, row in new_data.iterrows():
                    path_to_save = csv_file_path if csv_file_path else self.csv_file_path
                    self.rul_instance.update_measurements(current_time, row, save_to_csv, path_to_save)
            else:
                print("No new data available.")
            self.last_data_check_time = current_time
    

    def publish_rul_updates(self, all_rul_values):
        rul_values_blade_openfast = all_rul_values['rul_values_blade_openfast']
        rul_values_tower_openfast = all_rul_values['rul_values_tower_openfast']
        
        self.pub_socket.send_json({"rul_values_blade_openfast": rul_values_blade_openfast})
        self.pub_socket.send_json({"rul_values_tower_openfast": rul_values_tower_openfast})


    def publish_prediction(self, Pred_B, t_pred, current_time, present_state_web, time_horizon, Pred_Delta_B, Pred_B_Buffered, RotSpeed, WE_Vw, VS_GenPwr):
        Pred_Delta_B = Pred_Delta_B * 180 / np.pi
        current_time = current_time - time_horizon - 1
        self.pub_socket.send_json({"Pred_B": Pred_B, "t_pred": t_pred, "present_state_web": present_state_web, "current_time": current_time,"Pred_Delta_B": Pred_Delta_B, "Pred_B_Buffered": Pred_B_Buffered, "RotSpeed": RotSpeed, "WE_Vw": WE_Vw,"VS_GenPwr": VS_GenPwr})
        if current_time % 10 == 0:
            print(f"Published prediction updates: Pred_B: {Pred_B}, t_pred: {t_pred}, present state: {present_state_web}, current time: {current_time}, Pred_Delta_B: {Pred_Delta_B}, Pred_B_Buffered: {Pred_B_Buffered}, RotSpeed: {RotSpeed}, WE_Vw: {WE_Vw}, VS_GenPwr: {VS_GenPwr}")

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
            "TMax": 1100,
            "wave_height": wave_height,
            "peak_period": peak_period,
            "wave_direction": 0,
            "WvDiffQTF": "False",
            "WvSumQTF": "False"
        }

        self.steady_wind = True
        if self.steady_wind:
            print("Setting options for steady wind")
            r.wind_case_opts.update({"wind_type": 1, "HWindSpeed": 12.5})
        else:
            print("Setting options for turbulent wind")
            r.wind_case_opts.update({"turb_wind_speed": "20"})

        r.wind_case_opts["WaveTMax"] = r.wind_case_opts["TMax"] + 200
        print("Final case options:", r.wind_case_opts)
        r.controller_params = {"LoggingLevel": 2, "DISCON": {"ZMQ_Mode": 1, "ZMQ_ID": 1, "ZMQ_UpdatePeriod": 0.0125}}
        r.run_FAST()


    def main(self):
        processes = [
            mp.Process(target=self.run_zmq, args=(self.logfile,)),
            mp.Process(target=self.sim_openfast_custom),
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

if __name__ == "__main__":
    controller = CombinedController()
    controller.main()
    real_time_server_instance = RealTimeServer_class()
