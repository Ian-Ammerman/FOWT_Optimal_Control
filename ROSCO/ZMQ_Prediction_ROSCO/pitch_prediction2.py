import zmq
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
from scipy.stats import linregress
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from vmod import p2v

# Setup ZMQ for receiving turbine data
def setup_zmq_subscriber_for_turbine_data(port="5557"):
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(f"tcp://localhost:{port}")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe to all messages
    return subscriber

def receive_turbine_data(subscriber):
    message = subscriber.recv_string()  # Receive data as a string
    data = json.loads(message)  # Convert string back to Python dict
    return data

# Setup ZMQ for sending predicted blade pitch
def setup_zmq_publisher_for_blade_pitch(port="5556"):
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind(f"tcp://*:{port}")
    return publisher

def send_predicted_blade_pitch(publisher, delta_B):
    message = json.dumps({"predicted_blade_pitch": delta_B})
    publisher.send_string(message)

# Load and prepare model and data
def prepare_model_and_data(MODEL_PATH, SCALER_PATH):
    model = p2v.MLSTM()
    model.load_model(MODEL_PATH, SCALER_PATH)
    scaler = MinMaxScaler(feature_range=(0, 1))
    return model, scaler

def preprocess_data(data, scaler):
    # This function should be adjusted based on how your data needs to be preprocessed
    # For simplicity, assuming 'data' is already in the required format
    scaled = scaler.fit_transform(data)
    return scaled

def main():
    sub_port = "5557"
    pub_port = "5556"
    MODEL_PATH = os.path.join("MLSTM_WRP", "models", "7dof_MLSTM_WRP_OPT_T20_FC2")
    SCALER_PATH = os.path.join("MLSTM_WRP", "scalers", "scaler.pkl")

    subscriber = setup_zmq_subscriber_for_turbine_data(sub_port)
    publisher = setup_zmq_publisher_for_blade_pitch(pub_port)
    
    model, scaler = prepare_model_and_data(MODEL_PATH, SCALER_PATH)

    while True:
        turbine_data = receive_turbine_data(subscriber)
        # Preprocess the received data
        preprocessed_data = preprocess_data(turbine_data, scaler)
        
        # Predict using the preprocessed data
        prediction = model.predict(preprocessed_data)
        
        # Assume the prediction includes the blade pitch in the first position
        predicted_blade_pitch = prediction[0]
        
        # Send the predicted blade pitch
        send_predicted_blade_pitch(publisher, predicted_blade_pitch)

if __name__ == "__main__":
    main()
