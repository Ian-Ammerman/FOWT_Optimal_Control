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
import optuna
from optuna.visualization import plot_rank
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from vmod import p2v, get_psd

def objective(trial):
    nm = trial.suggest_float('n/m', 0.25, 2.5)
    waveshift_to_n = trial.suggest_float('waveshift', -2.0, 2.0)
    hidden_layer = 1
    neuron_number = 100
    epochs = 200
    batch_time = 47
    timestep = 1.0

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
    waveshift = int(n * waveshift_to_n)

    # Concatenate DOF with wind and wave data
    df = dataset_interpolated_inittransremoved[dof]
    wave_past = dataset_interpolated_inittransremoved["waveStaff5"]
    wind_past = winddata_interpolated_inittransremoved["wind"]
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
                               past_wind=past_wind, future_wind=future_wind, past_wave=past_wave,
                               future_wave=future_wave)
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

    rmse = np.zeros(len(labels))
    for i, label in enumerate(labels):
        label_index = label - 1
        rmse[i] = np.sqrt(mean_squared_error(original_test_Y[:, label_index], predicted_Y_wrp[:, label_index]))
        value_range = np.max(original_test_Y[:, label_index]) - np.min(original_test_Y[:, label_index])
        rmse[i] = rmse[i] / value_range


    return rmse.mean()
"""
In this example:
- optimize MLSTM-WRP on Focal Campaign 4 assuming the model has future information of both wind and wave
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

study_name = "03b_FC4_OPT"
study = optuna.create_study(study_name=study_name,
                            direction="minimize",
                            sampler=optuna.samplers.TPESampler(n_startup_trials=50, multivariate=True, group=True))
study.optimize(objective, n_trials=1000, gc_after_trial=True)

fig = plot_rank(study)