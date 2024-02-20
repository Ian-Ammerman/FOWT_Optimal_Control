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
- Perform data pre-processing and cleaning
- Concatenate DOF of interest with wave data
- Optimize MLSTM-WRP for a specified time horizon using optuna
"""

TEST_NUM = 1
DATA_INPUT_FILE = os.path.join("MLSTM_WRP", "Data", "FC2_URI", "S12_merged_10Hz_FS.csv")
TIME_HORIZON = 20

if not os.path.exists(os.path.join("figures", f"{TEST_NUM}")):
    os.makedirs(os.path.join("figures", f"{TEST_NUM}"))

# Data pre-processing
data = p2v.PreProcess(DATA_INPUT_FILE)
data.nan_check()
correlation_matrix = data.idle_sensors_check()
dataset = data.dataset

# LSTM
dof = ["PtfmTDX", "PtfmTDZ", "PtfmRDY", "leg1MooringForce", "leg2MooringForce", "leg3MooringForce", "towerBotMy"]
dof_with_units = ["surge [m]", "heave [m]", "pitch [deg]",
                  "cable 1 tension [kN]", "cable 2 tension [kN]", "cable 3 tension [kN]",
                  "fore-aft tower bending moment [kN-m]"]

nm = 0.588482
hidden_layer = 1
neuron_number = 96
epochs = 60
batch_time = 53
timestep = 1.0

new_time_range = np.arange(dataset['Time'].min(), dataset['Time'].max(), timestep)
dataset_interpolated = pd.DataFrame(new_time_range, columns=['Time'])

# Interpolate each column separately
for col in dataset.columns:
    if col != 'Time':  # Skip the 'Time' column
        dataset_interpolated[col] = np.interp(new_time_range, dataset['Time'], dataset[col])

# Compute n, m, and batch size
m = int(np.round(TIME_HORIZON / timestep, 0))  # corresponding to TIME_HORIZON
n = int(np.round(nm * m))
batch_size = int(np.round(batch_time / timestep, 0))

# Concatenate DOF with wave data
df = dataset_interpolated[dof]
wave_past = dataset_interpolated["waveStaff5"]
df_wrp = pd.concat([df, wave_past], axis=1).values

# Normalize the dataframe
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df_wrp)

# Create the supervised data
supervised_data = data.series_to_supervised(
    scaled,
    wind_var_number=None,
    wave_var_number=len(dof)+1,
    n_in=n,
    n_out=m,
    wind_predictor=False,
    wave_predictor=True)

# Train_Test ratio
train_ratio = 0.50
valid_ratio = 0.25

# Build, compile, and fit
past_wind = future_wind = False
past_wave = future_wave = True

features = list(np.arange(1, len(dof) + 1, 1))
labels = list(np.arange(1, len(dof) + 1, 1))

mlstm_wrp = p2v.MLSTM()
mlstm_wrp.split_train_test(supervised_data, train_ratio, valid_ratio, past_timesteps=n, future_timesteps=m,
                           features=features, labels=labels,
                           past_wind=past_wind, future_wind=future_wind, past_wave=past_wave, future_wave=future_wave)
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


fig = plt.figure(figsize=(12, 24))
gs = gridspec.GridSpec(len(dof), 2, width_ratios=[3, 1])  # 3:1 width ratio
conversion = [1, 1, 1, 1e-3, 1e-3, 1e-3, 1e-3]

for i, label in enumerate(labels):
    label_index = label - 1
    slope_wrp, intercept_wrp, r_value_wrp, _, _ = linregress(original_test_Y[:, label_index]*conversion[i], predicted_Y_wrp[:, label_index]*conversion[i])

    # Time domain data plots on the left side
    ax0 = plt.subplot(gs[i, 0])
    ax0.plot(original_test_Y[:, label_index]*conversion[i], label='exact', color='black')
    ax0.plot(predicted_Y_wrp[:, label_index]*conversion[i], label='MLSTM-WRP', color='red', linestyle='--')
    ax0.set_xlabel('t [s]')
    ax0.set_ylabel(f"{dof_with_units[label_index]}")
    ax0.set_xlim((5000, 5250))
    ax0.legend(loc='upper right')
    ax0.grid()

    # R plots on the right side (square plots)
    ax1 = plt.subplot(gs[i, 1], aspect='equal')  # aspect='equal' makes the plot square
    x_range = np.linspace(min(original_test_Y[:, label_index]*conversion[i]), max(predicted_Y_wrp[:, label_index]*conversion[i]), 100)
    ax1.plot(x_range, 1.0 * x_range + 0.0, color='black',
             label="exact")
    ax1.plot(x_range, slope_wrp * x_range + intercept_wrp, color='red', linestyle='--',
             label=f"MLSTM-WRP - $R^2$={np.round(r_value_wrp ** 2, 2)}")
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.legend(loc='lower right')
    ax1.grid()

plt.tight_layout()
plt.savefig(fr".\figures\{TEST_NUM}\TD_test_n_{n}_m_{m}_WRP_comp_02_before_fixing_converted.pdf", format="pdf")

# save the model
mlstm_wrp.save_model(os.path.join("MLSTM_WRP", "models", "7dof_MLSTM_WRP_OPT_T20_FC2"),
                     os.path.join("MLSTM_WRP", "scalers", "scaler.pkl"))
