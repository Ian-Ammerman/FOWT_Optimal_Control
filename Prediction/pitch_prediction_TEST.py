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



# Correct path to your Excel file
wave_data_path = '/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO/wave_data/wavedata.xlsx'

# Use read_excel to load Excel files
wave_data = pd.read_excel(wave_data_path, parse_dates=['t'])

def get_wave_data_for_prediction(current_time, lead_time=20):
    print("Reading wave elevation data")
    global wave_data_path
    """
    Fetches a slice of wave data leading the current time by a specified duration.

    :param current_time: datetime, The current timestamp.
    :param lead_time: int, The lead time in seconds for the wave data.
    :return: DataFrame, A slice of the wave data.
    """
    # Convert lead_time to a timedelta
    lead_timedelta = timedelta(seconds=lead_time)
    
    # Calculate the prediction time
    prediction_time = current_time + lead_timedelta
    
    # Load the data, ensuring 't' is in seconds and interpreted as floats
    wave_data = pd.read_excel(wave_data_path, decimal=',')
    wave_data['t'] = pd.to_numeric(wave_data['t'], errors='coerce')
    
    # Assume the 't' column represents seconds from the start and convert to datetime
    start_time = pd.Timestamp('2024-02-16 22:17:00')  # Replace with your actual start time
    wave_data['t'] = start_time + pd.to_timedelta(wave_data['t'], unit='s')
    
    # Now 't' is a datetime Series
    wave_data.sort_values('t', inplace=True)  # Sort by 't' to use searchsorted

    # Find the closest timestamp in the wave data to the prediction time
    closest_timestamp_index = wave_data['t'].searchsorted(prediction_time, side='left')
    
    # Handle edge case where prediction_time is beyond the last timestamp in wave_data
    if closest_timestamp_index >= len(wave_data):
        closest_timestamp_index = len(wave_data) - 1

    # Assuming you want a single row closest to the prediction time
    wave_data_slice = wave_data.iloc[closest_timestamp_index - 1: closest_timestamp_index]
    
    return wave_data_slice


def simulate_prediction(wave_data):
    # print("Simulating prediction model...")
    # This is a placeholder function that simulates prediction
    return random.uniform(-5, 5)  # Simulated delta_B value

def send_delta_B(publisher, delta_B, topic="delta_B"):
    """
    Publishes the delta_B value under a specified topic.
    """
    message = json.dumps({'delta_B': delta_B})
    full_message = f"{topic} {message}"  # Combine topic and message
    publisher.send_string(full_message)
    # print(f"PUBLISHED '{full_message}'")

def main():
    wave_data = pd.read_excel(wave_data_path, parse_dates=['t'])
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
