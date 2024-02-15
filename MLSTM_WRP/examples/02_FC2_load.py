from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from vmod import p2v

"""
In this example:
- load MLSTM-WRP on Focal Campaign 2
- test it on user-defined data
"""

TEST_NUM = 2
MODEL_PATH = os.path.join("MLSTM_WRP", "models", "7dof_MLSTM_WRP_OPT_T20_FC2")
SCALER_PATH = os.path.join("MLSTM_WRP", "scalers", "scaler.pkl")
DATA_INPUT_FILE = os.path.join("MLSTM_WRP", "Data", "FC2_URI", "S12_merged_10Hz_FS.csv")
TIME_HORIZON = 20

# load the model
model = p2v.MLSTM()
model.load_model(MODEL_PATH, SCALER_PATH)

# perform data pre-processing and cleaning
data = p2v.PreProcess(data_input_file=DATA_INPUT_FILE)
data.nan_check()
correlation_matrix = data.idle_sensors_check()

# Define LSTM Parameters
dof = ["PtfmTDX", "PtfmTDZ", "PtfmRDY", "leg1MooringForce", "leg2MooringForce", "leg3MooringForce", "towerBotMy"]
dof_with_units = ["surge [m]", "heave [m]", "pitch [deg]",
                  "cable 1 tension [kN]", "cable 2 tension [kN]", "cable 3 tension [kN]",
                  "fore-aft tower bending moment [kN-m]"]
dof_with_units_psd = [r"surge $[m/Hz^2]$", r"heave $[m/Hz^2]$", r"pitch $[deg/Hz^2]$",
                      r"cable 1 tension $[kN/Hz^2]$", r"cable 2 tension $[kN/Hz^2]$", r"cable 3 tension $[kN/Hz^2]$",
                      r"fore-aft tower bending moment $[kN-m/Hz^2]$"]
conversion = [1, 1, 1, 1e-3, 1e-3, 1e-3, 1e-3]

nm = 0.588482
timestep = 1.0
data.time_interpolator(timestep)
m = int(np.round(TIME_HORIZON / timestep, 0))  # corresponding to TIME_HORIZON
n = int(np.round(nm * m))
passing_time = np.concatenate((np.arange(-n, 0), np.linspace(0, m - 1, n)))

df = data.convert_extract(dof, conversion)
wavedata = data.dataset["waveStaff5"]
df_wrp = pd.concat([df, wavedata], axis=1).values

# Normalize the dataframe
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df_wrp)

# Supervise the data
features = list(np.arange(1, len(dof) + 1, 1))
labels = list(np.arange(1, len(dof) + 1, 1))
supervised_data = data.series_to_supervised(
    scaled,
    wind_var_number=None,
    wave_var_number=len(dof)+1,
    n_in=n,
    n_out=m,
    wind_predictor=False,
    wave_predictor=True)

past_wind = future_wind = False
past_wave = future_wave = True
input_columns = model.extract_input_columns(
    columns=supervised_data.columns,
    features=features,
    past_timesteps=n,
    past_wind=past_wind,
    future_wind=future_wind,
    past_wave=past_wave,
    future_wave=future_wave)
output_columns = model.extract_output_columns(
    columns=supervised_data.columns,
    labels=labels,
    future_timesteps=m)
num_features = len(features) + (1 if past_wind else 0) + (1 if future_wind else 0) + \
                       (1 if past_wave else 0) + (1 if future_wave else 0)
input_super_data = supervised_data[input_columns]
output_super_data = supervised_data[output_columns]
test_X = input_super_data.values
test_X = test_X.reshape((test_X.shape[0], n, num_features))
test_Y = output_super_data.values
# Run the predictive model
yhat = model.model.predict(test_X)

# Unscaling original data
num_original_features = df_wrp.shape[1]
dummy_array = np.zeros((test_X.shape[0], num_original_features))
dummy_array[:, :len(dof)] = test_Y
reversed_array = scaler.inverse_transform(dummy_array)
original_test_Y = reversed_array[:, :len(dof)]

# Unscaling predicted data
dummy_array = np.zeros((yhat.shape[0], num_original_features))
dummy_array[:, :len(dof)] = yhat
reversed_array = scaler.inverse_transform(dummy_array)
predicted_Y_wrp = reversed_array[:, :len(dof)]

R2 = np.zeros(len(labels))
for i, label in enumerate(labels):
    label_index = label - 1
    _, _, r_value_wrp, _, _ = linregress(original_test_Y[:, label_index],
                                         predicted_Y_wrp[:, label_index])
    R2[i] = r_value_wrp ** 2

print(f"R squared values are: {R2}")