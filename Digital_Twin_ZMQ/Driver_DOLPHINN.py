import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import multiprocessing as mp
from rosco.toolbox.ofTools.case_gen import CaseLibrary as cl
from rosco.toolbox.ofTools.case_gen.run_FAST import run_FAST_ROSCO
from rosco.toolbox.control_interface import wfc_zmq_server
from Digital_Twin_ZMQ.Blade_Pitch_Prediction.pitch_prediction import PredictionClass
from Digital_Twin_ZMQ.Blade_Pitch_Prediction.buffer_delta_B import buffer
import yaml

class bpcClass:
    def __init__(self, prediction_instance):
        self.prediction_instance = prediction_instance

        # directories
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.rosco_dir = os.path.dirname(self.this_dir)
        self.outputs = os.path.join('Sim_Outputs')
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

        # Specify path and load trained DOLPHINN model (Must contain BlPitchCMeas)
        DOLPHINN_PATH = os.path.join("Digital_Twin_ZMQ", "Blade_Pitch_Prediction", "DOLPHINN", "saved_models", "th10_Hs3_1_Tp8_U12Steady", "wave_model")
        config_file_path = os.path.join(DOLPHINN_PATH, 'config.yaml')
        with open(config_file_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        time_horizon = config_data['time_horizon']

        # Set to True for real time prediction plotting
        plot_figure = True

        # Get prediction and predicted time
        Pred_B, t_pred = self.prediction_instance.run_simulation(current_time, measurements, DOLPHINN_PATH, plot_figure, time_horizon)

        # Buffer duration
        buffer_duration = time_horizon
        # Buffer prediction until optimal time to send offset to ROSCO

        Pred_Delta_B = buffer(Pred_B, t_pred, current_time, measurements, buffer_duration)
        # Pred_Delta_B = 0.0
        YawOffset = 0.0

        # Set control setpoints
        setpoints = {}
        setpoints["ZMQ_YawOffset"] = YawOffset
        setpoints['ZMQ_PitOffset(1)'] = Pred_Delta_B
        setpoints['ZMQ_PitOffset(2)'] = Pred_Delta_B
        setpoints['ZMQ_PitOffset(3)'] = Pred_Delta_B

        return setpoints

    
    def sim_openfast(self):
        # Create an instance of the FAST simulation with ROSCO controller
        r = run_FAST_ROSCO()
        r.tuning_yaml = "IEA15MW_FOCAL.yaml"
        r.wind_case_fcn = cl.power_curve
        r.wind_case_opts = {"TMax": 2000}
        run_dir = os.path.join(self.outputs, "DOLPHINN_TESTING")
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
