import os
import matplotlib.pyplot as plt
from rosco.toolbox.inputs.validation import load_rosco_yaml
from rosco.toolbox.utilities import write_DISCON
from rosco.toolbox.control_interface import wfc_zmq_server
from rosco.toolbox.ofTools.case_gen import CaseLibrary as cl
from rosco.toolbox.ofTools.case_gen.run_FAST import run_FAST_ROSCO
from rosco.toolbox.ofTools.fast_io import output_processing
import numpy as np
import multiprocessing as mp

# Simulation parameters
TIME_CHECK = 30
DESIRED_YAW_OFFSET = 20
DESIRED_PITCH_OFFSET = np.deg2rad(2) * np.sin(0.1 * TIME_CHECK) + np.deg2rad(2)

# Directories
this_dir = os.path.dirname(os.path.abspath(__file__))
outputs = os.path.join(this_dir, 'bpc_outputs')
os.makedirs(outputs, exist_ok=True)

# ZeroMQ control server
def run_zmq(logfile=None):
    network_address = "tcp://*:5555"
    server = wfc_zmq_server(network_address, timeout=60.0, verbose=False, logfile=logfile)
    server.wfc_controller = wfc_controller
    server.runserver()

# Wind farm control logic
def wfc_controller(id, current_time, measurements):
    if current_time <= 10.0:
        yaw_setpoint = 0.0
    else:
        yaw_setpoint = DESIRED_YAW_OFFSET

    col_pitch_command = np.deg2rad(2) * np.sin(0.1 * current_time) + np.deg2rad(2) if current_time >= 10.0 else 0.0

    setpoints = {
        'ZMQ_TorqueOffset': 0.0,
        'ZMQ_YawOffset': yaw_setpoint,
        'ZMQ_PitOffset(1)': col_pitch_command,
        'ZMQ_PitOffset(2)': col_pitch_command,
        'ZMQ_PitOffset(3)': col_pitch_command
    }
    return setpoints

# OpenFAST simulation with ROSCO and ZeroMQ
def sim_openfast_rosco():
    # Load YAML configuration
    tune_dir = os.path.join(this_dir, 'Tune_Cases')
    parameter_filename = os.path.join(tune_dir, 'IEA15MW_FOCAL.yaml')
    inps = load_rosco_yaml(parameter_filename)

    # Set up ROSCO/OpenFAST simulation
    r = run_FAST_ROSCO()
    r.tuning_yaml = parameter_filename
    r.wind_case_fcn = cl.wind_input_test  # Specify wind case function if needed
    r.controller_params = {
        "LoggingLevel": 2,
        "DISCON": {
            "ZMQ_Mode": 1,
            "ZMQ_ID": 1,
            "ZMQ_UpdatePeriod": 0.025
        }
    }

    # Define simulation directory
    run_dir = os.path.join(outputs, '17_ZeroMQ')
    os.makedirs(run_dir, exist_ok=True)
    r.save_dir = run_dir

    # Execute OpenFAST simulation
    r.run_FAST()

if __name__ == "__main__":
    logfile = os.path.join(outputs, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    p0 = mp.Process(target=run_zmq, args=(logfile,))
    p1 = mp.Process(target=sim_openfast_rosco)

    p0.start()
    p1.start()

    p0.join()
    p1.join()

    # Process and display simulation results
    op = output_processing()
    debug_file = os.path.join(run_dir, 'IEA15MW_FOCAL.RO.dbg2')
    local_vars = op.load_fast_out([debug_file], tmin=0)

    # Example plot of yaw and pitch offsets over time
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(local_vars[0]['Time'], local_vars[0]['ZMQ_YawOffset'])
    axs[1].plot(local_vars[0]['Time'], local_vars[0]['ZMQ_PitOffset'])
    plt.show()

    # Spot check input at time = TIME_CHECK
    ind_check = local_vars[0]['Time'] == TIME_CHECK
    np.testing.assert_almost_equal(local_vars[0]['ZMQ_YawOffset'][ind_check], DESIRED_YAW_OFFSET)
    np.testing.assert_almost_equal(local_vars[0]['ZMQ_PitOffset'][ind_check], DESIRED_PITCH_OFFSET, decimal=3)


