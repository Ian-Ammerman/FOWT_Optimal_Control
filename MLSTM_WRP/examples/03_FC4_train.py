import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import os
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from vmod import p2v, get_psd

"""
In this example:
- train MLSTM-WRP on Focal Campaign 4 assuming the model has future information of both wind and wave
"""

TEST_NUM = 3
DATA_INPUT_FILE = os.path.join("MLSTM_WRP", "Data", "FC4", "windwave.csv")
WIND_DATA_INPUT_FILE = os.path.join("MLSTM_WRP", "Data", "FC4", "W02_winddata.csv")
TIME_HORIZON = 20

if not os.path.exists(os.path.join("figures", f"{TEST_NUM}")):
    os.makedirs(os.path.join("figures", f"{TEST_NUM}"))

# Data pre-processing
data = p2v.PreProcess(data_input_file=DATA_INPUT_FILE)
data.nan_check()
data.filter(direction="low", freq_cutoff=1)  # cut-off frequency at 1Hz
correlation_matrix = data.idle_sensors_check()
dataset = data.dataset

# Import wind data
winddata = pd.read_csv(WIND_DATA_INPUT_FILE)

# LSTM
dof = ["Surge", "Heave", "Pitch", "rotorTorque", "genSpeed", "leg3MooringForce", "accelNacelleAx", "towerBotMy"]
dof_with_units = ["surge [m]", "heave [m]", "pitch [deg]",
                  "rotor torque [kN.m]", "generator speed [rad/s]",
                  "line 3 tension [kN]",
                  "Nacelle Acceleration X [m/s^2]", "fore-aft tower bending moment [kN-m]"]
dof_with_units_f = ["surge [$m^2/Hz$]", "heave [$m^2/Hz$]", "pitch [$deg^2/Hz$]",
                    "rotor torque [$kN.m^2/Hz$", "generator speed [$(rad/s)^2/Hz$",
                    "tension [$kN^2/Hz$]",
                    "nacelle acceleration X [$(m/s)^2/Hz$]", "fore-aft tower bending moment [$(kN-m)^2/Hz$]"]
conversion = [1, 1, 1, 1e-3, 1, 1e-3, 1, 1e-3]
nm = 0.39
hidden_layer = 1
neuron_number = 100
epochs = 200
batch_time = 47
timestep = 0.73

new_time_range = np.arange(dataset['Time'].min(), dataset['Time'].max(), timestep)
dataset_interpolated = pd.DataFrame(new_time_range, columns=['Time'])
winddata_interpolated = pd.DataFrame(new_time_range, columns=['wind'])

# Interpolate each column separately
for col in dataset.columns:
    if col != 'Time':  # Skip the 'Time' column
        dataset_interpolated[col] = np.interp(new_time_range, dataset['Time'], dataset[col])

winddata_interpolated['wind'] = np.interp(new_time_range, winddata['time'], winddata['wind_speed'])

# Cut the initial transient from the simulation
time_cut = 1000
idx_cut = np.argmin(np.abs(time_cut - dataset_interpolated["Time"]))
dataset_interpolated_inittransremoved = dataset_interpolated.iloc[idx_cut:]
winddata_interpolated_inittransremoved = winddata_interpolated.iloc[idx_cut:]

# Compute n, m, and batch size
m = int(np.round(TIME_HORIZON / timestep, 0))  # corresponding to TIME_HORIZON
n = int(np.round(nm * m))
batch_size = int(np.round(batch_time / timestep, 0))

# Concatenate DOF with wind and wave data
df = dataset_interpolated_inittransremoved[dof]
wave_past = dataset_interpolated_inittransremoved["waveStaff5"]
wind_past = winddata_interpolated_inittransremoved["wind"]
df_wrp = pd.concat([df, wind_past, wave_past], axis=1).values

# Normalize the dataframe
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df_wrp)

# Create the supervised data
supervised_data = data.series_to_supervised(
    scaled,
    wind_var_number=len(dof) + 1,
    wave_var_number=len(dof) + 2,
    n_in=n,
    n_out=m,
    wind_predictor=True,
    wave_predictor=True)
# Train_Test ratio
train_ratio = 0.50
valid_ratio = 0.25

# Build, compile, and fit
features = list(np.arange(1, len(dof) + 1, 1))
labels = list(np.arange(1, len(dof) + 1, 1))

past_wind = False
future_wind = False
past_wave = True
future_wave = True

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
rmse = np.zeros(len(labels))
for i, label in enumerate(labels):
    label_index = label - 1
    _, _, r_value_wrp, _, _ = linregress(original_test_Y[:, label_index],
                                         predicted_Y_wrp[:, label_index])
    mae[i] = mean_absolute_error(original_test_Y[:, label_index], predicted_Y_wrp[:, label_index])
    rmse[i] = np.sqrt(mean_squared_error(original_test_Y[:, label_index], predicted_Y_wrp[:, label_index]))
    value_range = np.max(original_test_Y[:, label_index]) - np.min(original_test_Y[:, label_index])
    rmse[i] = rmse[i] / value_range
    R2[i] = r_value_wrp ** 2

fig = plt.figure(figsize=(12, 20))
gs = gridspec.GridSpec(len(dof), 2, width_ratios=[3, 1])  # 3:1 width ratio

for i, label in enumerate(labels):
    label_index = label - 1
    slope_wrp, intercept_wrp, r_value_wrp, _, _ = linregress(original_test_Y[:, label_index] * conversion[i],
                                                             predicted_Y_wrp[:, label_index] * conversion[i])

    # Time domain data plots on the left side
    ax0 = plt.subplot(gs[i, 0])
    ax0.plot(original_test_Y[:, label_index] * conversion[i], label='Experiment', color='black')
    ax0.plot(predicted_Y_wrp[:, label_index] * conversion[i], label='MLSTM-WRP', color='red', linestyle='--')
    ax0.set_xlabel('t [s]')
    ax0.set_ylabel(f"{dof_with_units[label_index]}")
    ax0.set_xlim((2000, 2250))
    if label == labels[0]:
        ax0.legend(loc='upper right')
    ax0.grid()

    # R plots on the right side (square plots)
    ax1 = plt.subplot(gs[i, 1], aspect='equal')  # aspect='equal' makes the plot square
    x_range = np.linspace(min(original_test_Y[:, label_index] * conversion[i]),
                          max(predicted_Y_wrp[:, label_index] * conversion[i]), 100)
    ax1.plot(x_range, 1.0 * x_range + 0.0, color='black')
    ax1.plot(x_range, slope_wrp * x_range + intercept_wrp, color='red', linestyle='--',
             label=f"$R^2$={np.round(r_value_wrp ** 2, 2)}")
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.legend(loc='lower right')
    ax1.grid()

plt.tight_layout()
plt.savefig(os.path.join("figures", f"{TEST_NUM}", "MLSTM_Wave.pdf"))

# save the model
mlstm_wrp.save_model(os.path.join("MLSTM_WRP", "models", f"FC4_OPT_MLSTM_WRP_{len(dof)}dof_T{TIME_HORIZON}"),
                     os.path.join("MLSTM_WRP", "scalers", "scaler.pkl"))

# Frequency plots
mpl.rcParams['font.family'] = 'Times New Roman'
time = np.array(dataset_interpolated_inittransremoved["Time"].iloc[-mlstm_wrp.test_X.shape[0]:] - dataset_interpolated_inittransremoved["Time"].iloc[
    -mlstm_wrp.test_X.shape[0]])
fig, ax = plt.subplots(4, 2, figsize=(12, 8))
ax = ax.flatten()
for i, label in enumerate(range(8)):
    label_index = label - 1

    F, PSD_exact = get_psd.get_PSD_limited(time, original_test_Y[:, i] - np.mean(original_test_Y[:, i]), 30, 0, 0.5)
    F_mlstm, PSD_mlstm = get_psd.get_PSD_limited(time, predicted_Y_wrp[:, i] - np.mean(original_test_Y[:, i]), 30, 0,
                                                 0.5)
    ax[i].plot(F, PSD_exact, label="experiment", color='black')
    ax[i].plot(F_mlstm, PSD_mlstm, label="MLSTM-WRP (Prediction Horizon=20s", color='red', linestyle='--')
    ax[i].set_xlabel('f [Hz]')
    ax[i].set_ylabel(dof_with_units_f[i])
    ax[i].set_xlim(0.005, 0.5)

    ax[i].grid()

ax[0].legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join("figures", f"{TEST_NUM}", "MLSTM_Wave_f.pdf"))
