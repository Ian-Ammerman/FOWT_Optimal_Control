"""
In this example:
- load the model
- perform data pre-processing and cleaning
- Define LSTM Parameters
- find wave characteristics based on certain checkpoint number and significant length
- Inside the Real-Time Loop:
    -
-
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from vmod import p2v, get_psd

TEST_NUM = 44
DATA_INPUT_FILE = os.path.join("MLSTM_WRP", "Data", "FC4", "windwave_test.csv")
TIME_HORIZON = 20

if not os.path.exists(os.path.join("figures", f"{TEST_NUM}")):
    os.makedirs(os.path.join("figures", f"{TEST_NUM}"))

# load the model
MODEL_PATH = os.path.join("MLSTM_WRP", "models", "FC4_OPT_MLSTM_WRP_8dof_T20_nm_0.39_dt_1.0_wvshftn_0.75")
SCALER_PATH = os.path.join("MLSTM_WRP", "scalers", "scaler.pkl")
model = p2v.MLSTM()
model.load_model(MODEL_PATH, SCALER_PATH)

# perform data pre-processing and cleaning
pre = p2v.PreProcess(DATA_INPUT_FILE)
pre.nan_check()
correlation_matrix = pre.idle_sensors_check()

# Define LSTM Parameters
dof = ["Surge", "Heave", "Pitch",
       "rotorTorque", "genSpeed",
       "leg3MooringForce",
       "accelNacelleAx", "towerBotMy"]
dof_with_units = ["surge [m]", "heave [m]", "pitch [deg]",
                  "rotor torque [kN.m]", "generator speed [rad/s]",
                  "line 3 tension [kN]",
                  "Nacelle Acceleration X [m/s^2]", "fore-aft tower bending moment [kN-m]"]

conversion = [1, 1, 1, 1e-3, 1, 1e-3, 1, 1e-3]

nm = 0.39
timestep = 1.0
waveshift_to_n = 0.75
pre.time_interpolator(timestep)
m = int(np.round(TIME_HORIZON / timestep, 0))  # corresponding to TIME_HORIZON
n = int(np.round(nm * m))
waveshift = int(n * waveshift_to_n)
passing_time = np.concatenate((np.arange(-n, 0), np.linspace(0, m - 1, n)))

# find wave characteristics based on certain checkpoint number and significant length
chkpnt_no = 100
significant_length = 1000  # in seconds
pre.dynamic_sig_char(chkpnt_no, significant_length)

df = pre.convert_extract(dof, conversion)
wavedata = pre.dataset["waveStaff5"]
df_wrp = pd.concat([df, wavedata], axis=1).values

scaler_b4_super = MinMaxScaler(feature_range=(0, 1))
scaler_b4_super.fit_transform(df_wrp)

dummy_supervised_data = pre.series_to_supervised(
    df_wrp,
    wind_var_number=None,
    wave_var_number=len(dof) + 1,
    n_in=n,
    n_out=m,
    waveshift=waveshift,
    wind_predictor=False,
    wave_predictor=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = pd.DataFrame(scaler.fit_transform(dummy_supervised_data), columns=dummy_supervised_data.columns)

Hss = np.zeros(len(dummy_supervised_data))
Tpp = np.zeros(len(dummy_supervised_data))
for i, t in enumerate(pre.dataset['Time'].iloc[:len(dummy_supervised_data)]):
    idx = np.where(~(t > pre.dynTi))[0][0]
    Hss[i] = pre.dynHs[idx]
    Tpp[i] = pre.dynTp[idx]

wrp_acc = pd.read_csv(os.path.join("MLSTM_WRP", "Data", "wrp", "wrp_accuracy.csv"))
alpha = 1.0
beta = 1.0

passing_time = np.concatenate((np.arange(-n, 0), np.linspace(0, m - 1, n)))

features = list(np.arange(1, len(dof) + 1, 1))
labels = list(np.arange(1, len(dof) + 1, 1))

past_wind = future_wind = False
past_wave = future_wave = True
input_columns = model.extract_input_columns(
    columns=dummy_supervised_data.columns,
    features=features,
    past_timesteps=n,
    past_wind=past_wind,
    future_wind=future_wind,
    past_wave=past_wave,
    future_wave=future_wave)
output_columns = model.extract_output_columns(
    columns=dummy_supervised_data.columns,
    labels=labels,
    future_timesteps=m)

num_features = len(features) + (1 if past_wind else 0) + (1 if future_wind else 0) + \
               (1 if past_wave else 0) + (1 if future_wave else 0)

# Real-Time Loop
def get_data_for_frame(frame, t):
    print(t[frame])
    idx = np.abs(pre.dataset['Time'] - t[frame]).argmin()
    history = 250
    y_size = history - waveshift + 1
    frame_time = np.arange(t[frame], t[frame] + y_size * timestep, timestep) - history
    if n * timestep <= pre.dataset['Time'][idx] < pre.dataset['Time'].iloc[-1] - m * timestep:
        # Create test X
        df_rt = df.iloc[idx-n-history: idx+m, :]
        wavedata_rt = wavedata[idx-n-history: idx+m]
        df_wrp_rt = pd.concat([df_rt, wavedata_rt], axis=1).values
        supervised_data = pre.series_to_supervised(
            df_wrp_rt,
            wind_var_number=None,
            wave_var_number=len(dof) + 1,
            n_in=n,
            n_out=m,
            waveshift=waveshift,
            wind_predictor=False,
            wave_predictor=True)

        # get current_wave (real)
        current_wave = supervised_data[f"wave(t+0.00)"]

        # use wave characteristics to add uncertainty to the wrp based on literature
        for t in passing_time:
            tTP = t / Tpp[idx]
            delta = np.interp(tTP, wrp_acc['x'] / alpha, wrp_acc['y'] * beta)
            supervised_data[f"wave(t{t:+.2f})"] += \
                np.random.uniform(1, 1, size=len(supervised_data)) * delta * Hss[idx]

        # get reconstructed wave
        reconstructed_wave = supervised_data[f"wave(t+{m-1:.2f})"]

        scaled = pd.DataFrame(scaler.transform(supervised_data), columns=supervised_data.columns)

        input_super_data = scaled[input_columns]

        test_X = input_super_data.values
        test_X = test_X.reshape((test_X.shape[0], n, num_features))
        yhat = model.model.predict(test_X)

        # Current states
        y = np.zeros((y_size, len(dof)))
        for dofi in range(len(dof)):
            y[:, dofi] = supervised_data[f"var{dofi + 1}(t+0.00)"]

        num_original_features = len(scaler_b4_super.data_max_)
        dummy_array = np.zeros((yhat.shape[0], num_original_features))
        dummy_array[:, :len(dof)] = yhat
        reversed_array = scaler_b4_super.inverse_transform(dummy_array)
        yhat = reversed_array[:, :len(dof)]

        # Unite mean values
        y_mean = df[dof].iloc[0:idx].mean(axis=0).values
        yhat_mean = yhat.mean(axis=0)
        yhat += y_mean - yhat_mean

    else:
        y = yhat = np.zeros((y_size, len(dof)))
        current_wave = reconstructed_wave = np.zeros(y_size)

    return frame_time, current_wave, reconstructed_wave, y, yhat


# Define an update function for animation
def update(frame):
    frame_time_new, current_wave_new, reconstructed_wave_new, y_new, yhat_new = get_data_for_frame(frame, t)

    # wave
    if frame == 0:
        frame_time_existing = []
        frame_time_updated = np.append(frame_time_existing, frame_time_new)
        current_wave_existing = []
        current_wave_updated = np.append(current_wave_existing, current_wave_new)
        reconstructed_wave_existing = []
        reconstructed_wave_updated = np.append(reconstructed_wave_existing, reconstructed_wave_new)
    else:
        frame_time_existing = lines[0].get_data()[0]
        frame_time_updated = np.append(frame_time_existing, frame_time_new[-1])
        current_wave_existing = lines[0].get_data()[1]
        current_wave_updated = np.append(current_wave_existing, current_wave_new.iloc[-1])
        reconstructed_wave_existing = lines_pred[0].get_data()[1]
        reconstructed_wave_updated = np.append(reconstructed_wave_existing, reconstructed_wave_new.iloc[-1])
        present_line[0].set_data([t[frame] - n, t[frame] - n], [np.min(current_wave_existing), np.max(current_wave_existing)])

    lines[0].set_data(frame_time_updated, current_wave_updated)
    lines_pred[0].set_data(frame_time_updated + m - 1, reconstructed_wave_updated)
    dot[0].set_offsets([[frame_time_updated[-1], current_wave_updated[-1]]])
    dot_pred[0].set_offsets([[frame_time_updated[-1] + m - 1, reconstructed_wave_updated[-1]]])
    ax[0].relim()
    ax[0].autoscale_view()
    ax[0].set_xlim((t[frame]-125, t[frame]+125))
    ax[0].set_xticks([])
    ax[0].set_ylabel('wave [m]')

    for dofi in np.arange(1, len(dofdof)+1):
        if frame == 0:
            y_existing = []
            y_updated = np.append(y_existing, y_new[:, dofi-1])
            yhat_existing = []
            yhat_updated = np.append(yhat_existing, yhat_new[:, dofi-1])
        else:
            y_existing = lines[dofi].get_data()[1]
            yhat_existing = lines_pred[dofi].get_data()[1]
            y_updated = np.append(y_existing, y_new[-1, dofi-1])
            yhat_updated = np.append(yhat_existing, yhat_new[-1, dofi-1])
            present_line[dofi].set_data([t[frame] - n, t[frame] - n] , [np.min(y_existing), np.max(y_existing)])
        lines[dofi].set_data(frame_time_updated, y_updated)
        lines_pred[dofi].set_data(frame_time_updated+m-1, yhat_updated)
        dot[dofi].set_offsets([[frame_time_updated[-1], y_updated[-1]]])
        dot_pred[dofi].set_offsets([[frame_time_updated[-1] + m - 1, yhat_updated[-1]]])
        ax[dofi].relim()
        ax[dofi].autoscale_view()
        ax[dofi].set_xlim((t[frame]-125, t[frame]+125))
        ax[dofi].set_xticks([])
        ax[dofi].set_ylabel(dof_with_units[dofi-1])

    return lines + lines_pred


# Preparing the plotting environment
dofdof = dof[0:3]
fig, ax = plt.subplots(len(dofdof)+1, 1, figsize=(10, 12))
lines = [axi.plot([], [], label='exact', color='black')[0] for axi in ax]
lines_pred = [axi.plot([], [], label='MLSTM-WRP', color='red', linestyle='--')[0] for axi in ax]
present_line = [axi.plot([], [], label='present', color='black', linestyle=':')[0] for axi in ax]
dot = [axi.scatter([], [], color='black') for axi in ax]
dot_pred = [axi.scatter([], [], color='red') for axi in ax]
[axi.grid() for axi in ax]
plt.legend()
# Create the animation
start_animation = 1700
end_animation = 1750
t = np.arange(start_animation, end_animation, timestep)
ani = FuncAnimation(fig, update, frames=np.arange(len(t)), blit=False)
ani.save(f'./figures/{TEST_NUM}/RT.gif', writer='pillow')