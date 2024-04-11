import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from vmod import p2v, get_psd

def run_MLSTM(start_simulation, end_simulation, timestep, TEST_NUM, MODEL_PATH, SCALER_PATH, data_frame_inputs, TIME_HORIZON):

    print("RUNNING MLSTM")

    if not os.path.exists(os.path.join("figures", f"{TEST_NUM}")):
        os.makedirs(os.path.join("figures", f"{TEST_NUM}"))

    # load the model
    model = p2v.MLSTM()
    model.load_model(MODEL_PATH, SCALER_PATH)
    print("MODEL LOADED SUCCESSFULLY")

    # perform data pre-processing on dummy data and cleaning
    data = p2v.PreProcess(data_input_file=data_frame_inputs)
    print("PREPROCESSING FINISHED SUCCESSFULLY")
    data.nan_check()
    correlation_matrix = data.idle_sensors_check()

    # LSTM
    dof = [ 'PtfmTDX', 'PtfmTDZ', 'PtfmRDY', 'GenTqMeas', 'RotSpeed', 'NacIMU_FA_Acc', 'PtfmRDY', 'PtfmRDZ']

    dof_with_units = ["surge [m]", "heave [m]", "pitch [deg]",
                    "rotor torque [kN.m]", "generator speed [rad/s]",
                    "line 3 tension [kN]",
                    "Nacelle Acceleration X [m/s^2]", "fore-aft tower bending moment [kN-m]"]

    conversion = [1, 1, 1, 1e-3, 1, 1e-3, 1, 1e-3]

    nm = 0.39
    timestep = 1.0
    waveshift_to_n = 0.75
    data.time_interpolator(timestep)
    m = int(np.round(TIME_HORIZON / timestep, 0))  # corresponding to TIME_HORIZON
    n = int(np.round(nm * m))
    waveshift = int(n * waveshift_to_n)
    passing_time = np.concatenate((np.arange(-n, 0), np.linspace(0, m - 1, n)))

    # find wave characteristics based on certain checkpoint number and significant length
    chkpnt_no = 100
    significant_length = 8  # in seconds
    data.dynamic_sig_char(chkpnt_no, significant_length)

    df = data.convert_extract(dof, conversion)
    # Data does not contain waveStaff5, replace with windwave_test.csv ?
    wavedata = data.dataset["waveStaff5"]
    df_wrp = pd.concat([df, wavedata], axis=1).values

    # Normalize the dataframe
    scaler_b4_super = MinMaxScaler(feature_range=(0, 1))
    scaler_b4_super.fit_transform(df_wrp)

    dummy_supervised_data = data.series_to_supervised(
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

    # Hs and Tp as a function of time
    Hss = np.zeros(len(dummy_supervised_data))
    Tpp = np.zeros(len(dummy_supervised_data))
    for i, t in enumerate(data.dataset['Time'].iloc[:len(dummy_supervised_data)]):
        idx = np.where(~(t > data.dynTi))[0][0]
        Hss[i] = data.dynHs[idx]
        Tpp[i] = data.dynTp[idx]

    # Adding error function onto the wave to simulate wave prediction model
    wrp_acc = pd.read_csv("/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO/MLSTM_WRP/Data/wrp/wrp_accuracy.csv")
    alpha = 1.0
    beta = 1.0
    
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

    # Real-Time Simulation
    start_simulation = 1700
    end_simulation = 2000
    T = np.arange(start_simulation, end_simulation, timestep)
    for i, t in enumerate(T):
        y, yhat = get_data_for_frame(t, data, model, dof, n, m, waveshift, scaler_b4_super, scaler, wavedata, timestep, df, input_columns, num_features)
        print(f"y = {y[i]}\n yhat = {yhat[i]}")

    
# Real-Time Loop
def get_data_for_frame(t, data, model, dof, n, m, waveshift, scaler_b4_super, scaler, wavedata, timestep, df, input_columns, num_features):
    print(f"t = {t}s")
    idx = np.abs(data.dataset['Time'] - t).argmin()
    y_size = waveshift
    if n * timestep <= data.dataset['Time'][idx] < data.dataset['Time'].iloc[-1] - m * timestep:
        # Create test X
        df_rt = df.iloc[idx-n-y_size: idx+m, :]
        wavedata_rt = wavedata[idx-n-y_size: idx+m]
        df_wrp_rt = pd.concat([df_rt, wavedata_rt], axis=1).values
        supervised_data = data.series_to_supervised(
            df_wrp_rt,
            wind_var_number=None,
            wave_var_number=len(dof) + 1,
            n_in=n,
            n_out=m,
            waveshift=waveshift,
            wind_predictor=False,
            wave_predictor=True)

        # use wave characteristics to add uncertainty to the wrp based on literature
        # for t in passing_time:
        #     tTP = t / Tpp[idx]
        #     delta = np.interp(tTP, wrp_acc['x'] / alpha, wrp_acc['y'] * beta)
        #     supervised_data[f"wave(t{t:+.2f})"] += \
        #         np.random.uniform(1, 1, size=len(supervised_data)) * delta * Hss[idx]

        scaled = pd.DataFrame(scaler.transform(supervised_data), columns=supervised_data.columns)

        input_super_data = scaled[input_columns]

        test_X = input_super_data.values
        test_X = test_X.reshape((test_X.shape[0], n, num_features))
        yhat = model.model.predict(test_X)

        # Current states
        y = np.zeros((y_size - waveshift + 1, len(dof)))
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

    return y, yhat
