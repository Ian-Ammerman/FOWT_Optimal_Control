import zmq
import json  # Assuming data is sent as JSON strings
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
from rosco.toolbox.control_interface import wfc_zmq_server
from sklearn.metrics import mean_absolute_error
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from vmod import p2v
import joblib
import logging
from datetime import datetime, timedelta
import random
import time
print("RUNNING PITCH_PREDICTION.PY")
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def setup_zmq_subscriber(port="5555", topic=""):
    print("Setting up subscriber (Port 5555)...")
    # Prepare our context and subscriber socket
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    # Connect to the publisher socket
    subscriber.connect(f"tcp://localhost:{port}")
    # Subscribe to the specified topic
    subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)
    return subscriber

def setup_zmq_publisher(port="5556"):
    print("Setting up publisher (Port 5556)...")
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind(f"tcp://*:{port}")
    return publisher


def receive_data(subscriber):
    try:
        topic, message = subscriber.recv_multipart()
        return json.loads(message.decode('utf-8'))
    except zmq.Again as e:
        print(f"Timeout occurred: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def send_delta_B(publisher, delta_B, topic="delta_B"):
    """
    Publishes the delta_B value under a specified topic.
    """
    message = json.dumps({'delta_B': delta_B})
    full_message = f"{topic} {message}"  # Combine topic and message
    publisher.send_string(full_message)
    # print(f"PUBLISHED '{full_message}'")

def main():
    port_publisher = "5556"  # Port to publish predictions
    publisher = setup_zmq_publisher(port=port_publisher)  # Setup ZeroMQ publisher
    
    delta_B_counter = 1  # Initialize the counter at 1
    
    try:
        while True:
            # Increment and publish the counter as delta_B
            send_delta_B(publisher, delta_B_counter, topic="delta_B")
            # print(f"Publishing delta_B: {delta_B_counter}")  # Optional: print the published value
            
            delta_B_counter += 1  # Increment the counter
            
            time.sleep(1)  # Add a delay to simulate time between predictions, adjust as needed
    except KeyboardInterrupt:
        print("Terminating publisher...")
        publisher.close()
        context.term()

if __name__ == "__main__":
    main()
