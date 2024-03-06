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


TEST_NUM = 5
KF_DATA_INPUT_FILE = os.path.join("MLSTM_WRP", "Data", "FC4", "KF_above_rated.csv")
wave_DATA_INPUT_FILE = os.path.join("MLSTM_WRP", "Data", "FC4", "WindWave_AboveRated_wave_file.Elev")
TIME_HORIZON = 20

if not os.path.exists(os.path.join("figures", f"{TEST_NUM}")):
    os.makedirs(os.path.join("figures", f"{TEST_NUM}"))

# Data pre-processing
data = p2v.PreProcess(data_input_file=KF_DATA_INPUT_FILE)
data.nan_check()
data.filter(direction="low", freq_cutoff=1)  # cut-off frequency at 1Hz
# data.filter(direction="high", freq_cutoff=0.05)  # cut-off frequency at 0.05Hz
correlation_matrix = data.idle_sensors_check()
dataset = data.dataset

wavedata = pd.read_csv(wave_DATA_INPUT_FILE, header=None, names=["Time", "Wave"])

# LSTM
dof = ["Surge", "Heave", "Pitch", "TwrBsMyt"]
dof_with_units = ["surge [m]", "heave [m]", "pitch [deg]",
                  "fore-aft tower bending moment [kN-m]"]
dof_with_units_f = ["surge [$m^2/Hz$]", "heave [$m^2/Hz$]", "pitch [$deg^2/Hz$]",
                    "fore-aft tower bending moment [$(kN-m)^2/Hz$]"]
conversion = [1, 1, 1, 1, 1e-3]

nm = 1.704
waveshift_to_n = 0.236
hidden_layer = 1
neuron_number = 100
epochs = 80
batch_time = 47
timestep = 1.0

# Time interpolation
new_time_range = np.arange(dataset['Time'].min(), dataset['Time'].max(), timestep)
dataset_interpolated = pd.DataFrame(new_time_range, columns=['Time'])
wavedata_interpolated = pd.DataFrame(new_time_range, columns=['Wave'])

# Interpolate each column separately
for col in dataset.columns:
    if col != 'Time':  # Skip the 'Time' column
        dataset_interpolated[col] = np.interp(new_time_range, dataset['Time'], dataset[col])

wavedata_interpolated['Wave'] = np.interp(new_time_range, wavedata['Time'], wavedata['Wave'])


# Cut the initial transient from the simulation
time_cut = 1000
idx_cut = np.argmin(np.abs(time_cut - dataset_interpolated["Time"]))
dataset_interpolated_inittransremoved = dataset_interpolated.iloc[idx_cut:]
wavedata_interpolated_inittransremoved = wavedata_interpolated.iloc[idx_cut:]

# Compute n, m, and batch size
m = int(np.round(TIME_HORIZON / timestep, 0))  # corresponding to TIME_HORIZON
n = int(np.round(nm * m))
batch_size = int(np.round(batch_time / timestep, 0))
waveshift = int(n * waveshift_to_n)

# Concatenate DOF with wind and wave data
df = dataset_interpolated_inittransremoved[dof]
wave_past = wavedata_interpolated_inittransremoved["Wave"]
# df_wrp = pd.concat([df, wind_past, wave_past], axis=1).values
df_wrp = pd.concat([df, wave_past], axis=1).values

# Normalize the dataframe
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df_wrp)


# Create the supervised data
supervised_data = data.series_to_supervised(
    scaled,
    wind_var_number=None,
    wave_var_number=len(dof) + 1,
    n_in=n,
    n_out=m,
    waveshift=waveshift,
    wind_predictor=False,
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
    ax0.set_xlim((1700, 2000))
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
# plt.savefig(os.path.join("figures", f"{TEST_NUM}", "MLSTM_Wave.pdf"))

# Frequency plots
mpl.rcParams['font.family'] = 'Times New Roman'
time = np.array(dataset_interpolated_inittransremoved["Time"].iloc[-mlstm_wrp.test_X.shape[0]:] - dataset_interpolated_inittransremoved["Time"].iloc[
    -mlstm_wrp.test_X.shape[0]])
fig, ax = plt.subplots(len(dof), 1, figsize=(12, 8))
ax = ax.flatten()
for i, label in enumerate(range(len(dof))):
    label_index = label - 1

    F, PSD_exact = get_psd.get_PSD_limited(time, original_test_Y[:, i] - np.mean(original_test_Y[:, i]), 30, 0, 0.5)
    F_mlstm, PSD_mlstm = get_psd.get_PSD_limited(time, predicted_Y_wrp[:, i] - np.mean(original_test_Y[:, i]), 30, 0,
                                                 0.5)
    ax[i].plot(F, PSD_exact, label="experiment", color='black')
    ax[i].plot(F_mlstm, PSD_mlstm, label=f"MLSTM-WRP (Prediction Horizon={TIME_HORIZON}s)", color='red', linestyle='--')
    ax[i].set_xlabel('f [Hz]')
    ax[i].set_ylabel(dof_with_units_f[i])
    ax[i].set_xlim(0.005, 0.5)

    ax[i].grid()

ax[0].legend(loc='upper right')
plt.tight_layout()
# plt.savefig(os.path.join("figures", f"{TEST_NUM}", "MLSTM_Wave_f.pdf"))