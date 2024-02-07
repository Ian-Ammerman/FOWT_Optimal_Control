import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
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
- Perform MLSTM-WRP on Focal Campaign 4 with wind-wave test (data are collected at 24Hz)
"""

TEST_NUM = 3
DATA_INPUT_FILE = os.path.join("MLSTM_WRP", "Data", "FC4", "windwave.csv")
WIND_DATA_INPUT_FILE = os.path.join("MLSTM_WRP", "Data", "FC4", "W02_winddata.csv")
TIME_HORIZON = 20

if not os.path.exists(os.path.join("figures", f"{TEST_NUM}")):
    os.makedirs(os.path.join("figures", f"{TEST_NUM}"))

# Data pre-processing
data = p2v.PreProcess(DATA_INPUT_FILE)
data.nan_check()
data.filter(direction="low", freq_cutoff=1)  # cut-off frequency at 1Hz
correlation_matrix = data.idle_sensors_check()
dataset = data.dataset

# Import wind data
winddata = pd.read_csv(WIND_DATA_INPUT_FILE)

# Plot the heatmap
mpl.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(7, 7))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.savefig(os.path.join("figures", f"{TEST_NUM}", "correlation heatmap.pdf"), format="pdf")

# LSTM
dof = ["Surge", "Heave", "Pitch", "rotorTorque", "genSpeed", "leg1MooringForce", "accelNacelleAx", "towerBotMy"]
dof_with_units = ["surge [m]", "heave [m]", "pitch [deg]",
                  "rotor torque [kN.m]", "generator speed [rad/s]",
                  "cable 1 tension [kN]",
                  "Nacelle Acceleration X [m/s^2]", "fore-aft tower bending moment [kN-m]"]

conversion = [1, 1, 1, 1e-3, 1, 1e-3, 1, 1e-3]
nm = 0.588482
hidden_layer = 1
neuron_number = 96
epochs = 60
batch_time = 53
timestep = 1.00

new_time_range = np.arange(dataset['Time'].min(), dataset['Time'].max(), timestep)
dataset_interpolated = pd.DataFrame(new_time_range, columns=['Time'])
winddata_interpolated = pd.DataFrame(new_time_range, columns=['wind'])

# Interpolate each column separately
for col in dataset.columns:
    if col != 'Time':  # Skip the 'Time' column
        dataset_interpolated[col] = np.interp(new_time_range, dataset['Time'], dataset[col])

winddata_interpolated['wind'] = np.interp(new_time_range, winddata['time'], winddata['wind_speed'])

# Compute n, m, and batch size
m = int(np.round(TIME_HORIZON / timestep, 0))  # corresponding to TIME_HORIZON
n = int(np.round(nm * m))
batch_size = int(np.round(batch_time / timestep, 0))

# Concatenate DOF with wind and wave data
df = dataset_interpolated[dof]
wave_past = dataset_interpolated["waveStaff5"]
wind_past = winddata_interpolated["wind"]
df_wrp = pd.concat([df, wind_past, wave_past], axis=1).values

# Normalize the dataframe
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df_wrp)

# Create the supervised data
supervised_data = data.series_to_supervised(scaled, len(dof) + 2, n, m, wp=True)

# Train_Test ratio
train_ratio = 0.50
valid_ratio = 0.25


# Build, compile, and fit
features = list(np.arange(1, len(dof) + 1, 1))
labels = list(np.arange(1, len(dof) + 1, 1))

mlstm_wrp = p2v.MLSTM()
mlstm_wrp.split_train_test(supervised_data, train_ratio, valid_ratio, past_timesteps=n, future_timesteps=m,
                           features=features, labels=labels, past_wrp=True, future_wrp=True)
mlstm_wrp.build_and_compile_model(hidden_layer, neuron_number, len(labels), lr=0.001)
history = mlstm_wrp.model.fit(mlstm_wrp.train_X, mlstm_wrp.train_Y, epochs=epochs, batch_size=batch_size,
                              validation_data=(mlstm_wrp.valid_X, mlstm_wrp.valid_Y), verbose=2, shuffle=False)

# Test the models
test_predictions_wrp = mlstm_wrp.model.predict(mlstm_wrp.test_X)

# Unscaling original data
num_original_features = df_wrp.shape[1]
dummy_array = np.zeros((mlstm_wrp.test_Y.shape[0], num_original_features))
dummy_array[:, :len(dof)] = mlstm_wrp.test_Y
reversed_array = scaler.inverse_transform(dummy_array)
original_test_Y = reversed_array[:, :len(dof)]

# Unscaling predicted data
dummy_array = np.zeros((test_predictions_wrp.shape[0], num_original_features))
dummy_array[:, :len(dof)] = test_predictions_wrp
reversed_array = scaler.inverse_transform(dummy_array)
predicted_Y_wrp = reversed_array[:, :len(dof)]

R2 = np.zeros(len(labels))
mae = np.zeros(len(labels))
for i, label in enumerate(labels):
    label_index = label - 1
    _, _, r_value_wrp, _, _ = linregress(original_test_Y[:, label_index],
                                         predicted_Y_wrp[:, label_index])
    mae[i] = mean_absolute_error(original_test_Y[:, label_index], predicted_Y_wrp[:, label_index])
    R2[i] = r_value_wrp ** 2


fig = plt.figure(figsize=(12, 20))
gs = gridspec.GridSpec(len(dof), 2, width_ratios=[3, 1])  # 3:1 width ratio

for i, label in enumerate(labels):
    label_index = label - 1
    slope_wrp, intercept_wrp, r_value_wrp, _, _ = linregress(original_test_Y[:, label_index]*conversion[i], predicted_Y_wrp[:, label_index]*conversion[i])

    # Time domain data plots on the left side
    ax0 = plt.subplot(gs[i, 0])
    ax0.plot(original_test_Y[:, label_index]*conversion[i], label='Experiment', color='black')
    ax0.plot(predicted_Y_wrp[:, label_index]*conversion[i], label='MLSTM-WRP', color='red', linestyle='--')
    ax0.set_xlabel('t [s]')
    ax0.set_ylabel(f"{dof_with_units[label_index]}")
    ax0.set_xlim((2000, 2250))
    if label == labels[0]:
        ax0.legend(loc='upper right')
    ax0.grid()

    # R plots on the right side (square plots)
    ax1 = plt.subplot(gs[i, 1], aspect='equal')  # aspect='equal' makes the plot square
    x_range = np.linspace(min(original_test_Y[:, label_index]*conversion[i]), max(predicted_Y_wrp[:, label_index]*conversion[i]), 100)
    ax1.plot(x_range, 1.0 * x_range + 0.0, color='black')
    ax1.plot(x_range, slope_wrp * x_range + intercept_wrp, color='red', linestyle='--',
             label=f"$R^2$={np.round(r_value_wrp ** 2, 2)}")
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.legend(loc='lower right')
    ax1.grid()

plt.tight_layout()
plt.savefig(os.path.join("figures", f"{TEST_NUM}", "8dof_TD_test_n_12_m_20.pdf"))

# save the model
mlstm_wrp.save_model(os.path.join("MLSTM_WRP", "models", "8dof_MLSTM_WRP_OPT_T20_FC4"),
                     os.path.join("MLSTM_WRP", "scalers", "scaler.pkl"))
