import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from rosco.toolbox.control_interface import wfc_zmq_server
from rosco.toolbox.ofTools.case_gen import CaseLibrary as cl
from rosco.toolbox.ofTools.case_gen.run_FAST import run_FAST_ROSCO
from rosco.toolbox.ofTools.fast_io import output_processing
import subprocess
import zmq
import sys
import json
import threading
from multiprocessing import Value, Process
import time
print("RUNNING BLADE_PITCH_CONTROLLER.PY")

manager = mp.Manager()
delta_B_buffer = manager.list()
latest_delta_B = Value('d', 0.0)  # 'd' indicates a double precision float

DESIRED_YAW_OFFSET = [-10, 10]

#directories
this_dir            = os.path.dirname(os.path.abspath(__file__))
rosco_dir           = os.path.dirname(this_dir)
outputs     = os.path.join(this_dir,'bpc_outputs')
os.makedirs(outputs,exist_ok=True)

def run_zmq(logfile=None):
    # Start the server at the following address
    network_address = "tcp://*:5555"
    server = wfc_zmq_server(network_address, timeout=60.0, verbose=False, logfile=logfile)

    # Provide the wind farm control algorithm as the wfc_controller method of the server
    server.wfc_controller = wfc_controller

    # Run the server to receive measurements and send setpoints
    server.runserver()

def setup_delta_B_subscriber(port="5556", topic="delta_B"):
    """
    Setup a ZeroMQ subscriber to receive data on the given port and optional topic.
    """
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(f"tcp://localhost:{port}")
    if topic:
        subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
    else:
        subscriber.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages if no topic is specified
    return subscriber

def process_delta_B_with_delay():
    global delta_B_buffer, latest_delta_B
    while True:
        current_time = time.time()
        to_remove = []
        for item in delta_B_buffer:
            delta_B, timestamp = item
            if current_time - timestamp >= 20:  # 20 seconds delay
                latest_delta_B.value = delta_B
                to_remove.append(item)
        for item in to_remove:
            delta_B_buffer.remove(item)
        time.sleep(1)  # Sleep to prevent a tight loop

def wfc_controller(id, current_time, measurements):
    global latest_delta_B

    updated_delta_B = latest_delta_B.value

    if current_time <= 10.0:
        YawOffset = 0.0
    else:
        if id == 1:
            YawOffset = DESIRED_YAW_OFFSET[0]
        else:
            YawOffset = DESIRED_YAW_OFFSET[1]

    # Use latest_delta_B for blade pitch control
    col_pitch_command = updated_delta_B  # Assuming latest_delta_B is already in radians
    print("COL_PITCH_COMMAND (+0s):", col_pitch_command)
    
    setpoints = {}
    setpoints["ZMQ_YawOffset"] = YawOffset
    setpoints['ZMQ_PitOffset(1)'] = col_pitch_command
    setpoints['ZMQ_PitOffset(2)'] = col_pitch_command
    setpoints['ZMQ_PitOffset(3)'] = col_pitch_command
    return setpoints
    
def sim_openfast():
    global latest_delta_B

    # Create an instance of the FAST simulation with ROSCO controller
    r = run_FAST_ROSCO()
    r.tuning_yaml = "IEA15MW_FOCAL.yaml"
    r.wind_case_fcn = cl.power_curve
    r.wind_case_opts = {
        "U": [8],
        "TMax": 1000,
    }
    run_dir = os.path.join(outputs, "17b_zeromq_OF1")
    r.controller_params = {}
    r.controller_params["LoggingLevel"] = 2
    r.controller_params["DISCON"] = {}
    r.controller_params["DISCON"]["ZMQ_Mode"] = 1
    r.controller_params["DISCON"]["ZMQ_ID"] = 1
    r.save_dir = run_dir
    r.run_FAST()

def listen_for_delta_B():
    subscriber = setup_delta_B_subscriber("5556")
    print("Listening for delta_B values...")
    first_delta_B_time = None  # Initialize to None, indicating no delta_B received yet
    
    while True:
        try:
            full_message = subscriber.recv_string()
            topic, message_content = full_message.split(' ', 1)
            data = json.loads(message_content)
            current_time = time.time()  # Get current time in seconds
            
            # If first_delta_B_time is None, this is the first delta_B received
            if first_delta_B_time is None:
                first_delta_B_time = current_time
                print("First delta_B received.")
            else:
                # Calculate and print the time passed since the first delta_B was received
                time_passed = current_time - first_delta_B_time
                if time_passed < 21:
                    print(f"Time passed since first delta_B: {time_passed:.2f} seconds.")
            
            # Store delta_B with the current timestamp in the buffer
            delta_B_buffer.append((data['delta_B'], current_time))
            print("REAL TIME DELTA_B (+20s):", data['delta_B'])

        except Exception as e:
            print(f"Error receiving message: {e}")

if __name__ == "__main__":
    logfile = os.path.join(outputs, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    
    # Start the pitch_prediction.py as a subprocess with immediate output to the console
    pitch_prediction_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Prediction/pitch_prediction.py')
    pitch_prediction_process = subprocess.Popen(["python", pitch_prediction_script], stdout=sys.stdout, stderr=sys.stderr)
    print("Started pitch_prediction.py subprocess.")
    
    # Start the existing processes
    p1 = mp.Process(target=listen_for_delta_B)
    p2 = mp.Process(target=run_zmq, args=(logfile,))
    p3 = mp.Process(target=sim_openfast)
    p4 = mp.Process(target=process_delta_B_with_delay)

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    # Wait for multiprocessing processes to complete
    p1.join()
    p2.join()
    p3.join()
    p4.join()
