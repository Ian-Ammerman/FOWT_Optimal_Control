import yaml
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import numpy as np
import p2v
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error
import pickle
import yaml
import joblib
from tensorflow.keras.models import load_model


class DOLPHINN:
    def __init__(self):
        self.future_lower_lim = None
        self.past_lower_lim = None
        self.n = None
        self.m = None
        self.correlation_matrix = None
        self.prep = None
        self.wave_prediction = None
        self.wind_prediction = None
        self.valid_ratio = None
        self.train_ratio = None
        self.dropout = None
        self.lr = None
        self.timestep = None
        self.batch_time = None
        self.epochs = None
        self.neuron_number = None
        self.hidden_layer = None
        self.nm = None
        self.time_horizon = None
        self.conversion = None
        self.unit = None
        self.dof = None
        self.data_input_file = None
        self.scaler = None
        self.config_path = None

        self.mlstm_wrp = p2v.MLSTM()

    def save(self, directory):
        """Save the complete state of the class."""
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save configuration
        config_path = os.path.join(directory, 'config.yaml')
        self.save_config(config_path)

        # Save the model
        model_path = os.path.join(directory, 'model.keras')
        self.mlstm_wrp.model.save(model_path)

        # Save the scaler
        scaler_path = os.path.join(directory, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        # Save necessary attributes
        attributes_path = os.path.join(directory, 'attributes.pkl')
        with open(attributes_path, 'wb') as f:
            pickle.dump({
                'prep': self.prep,
            }, f)

    def load(self, directory):
        """Load the complete state of the class."""
        # Load configuration
        self.config_path = os.path.join(directory, 'config.yaml')
        self.load_config()

        # Load the model
        model_path = os.path.join(directory, 'model.keras')
        self.mlstm_wrp.model = load_model(model_path)

        # Load the scaler
        scaler_path = os.path.join(directory, 'scaler.pkl')
        self.scaler = joblib.load(scaler_path)

        # Load necessary attributes
        attributes_path = os.path.join(directory, 'attributes.pkl')
        with open(attributes_path, 'rb') as f:
            attributes = pickle.load(f)
            self.prep = attributes['prep']

    def load_config(self):
        """Load MLSTM configuration from a YAML file."""
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.data_input_file = config.get('training_dataset')
        self.dof = config.get('dof')
        self.unit = config.get('unit')
        self.conversion = config.get('conversion')
        self.time_horizon = config.get('time_horizon')
        self.nm = config.get('nm')
        self.hidden_layer = config.get('hidden_layer')
        self.neuron_number = config.get('neuron_number')
        self.epochs = config.get('epochs')
        self.batch_time = config.get('batch_time')
        self.timestep = config.get('timestep')
        self.lr = config.get('lr')
        self.dropout = config.get('dropout')
        self.train_ratio = config.get('train_ratio')
        self.valid_ratio = config.get('valid_ratio')
        self.wind_prediction = config.get('wind_prediction')
        self.wave_prediction = config.get('wave_prediction')

        # dependent attributes
        self.m = int(np.round(self.time_horizon / self.timestep, 0))  # corresponding to TIME_HORIZON
        self.n = int(np.round(self.nm * self.m))
        self.future_lower_lim = self.m

    def save_config(self, config_path):
        """Save the current configuration to a YAML file."""
        config = {
            'training_dataset': self.data_input_file,
            'dof': self.dof,
            'unit': self.unit,
            'conversion': self.conversion,
            'time_horizon': self.time_horizon,
            'nm': self.nm,
            'hidden_layer': self.hidden_layer,
            'neuron_number': self.neuron_number,
            'epochs': self.epochs,
            'batch_time': self.batch_time,
            'timestep': self.timestep,
            'lr': self.lr,
            'dropout': self.dropout,
            'train_ratio': self.train_ratio,
            'valid_ratio': self.valid_ratio,
            'wind_prediction': self.wind_prediction,
            'wave_prediction': self.wave_prediction
        }
        with open(config_path, 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False)

    def preprocessor(self):
        self.prep = p2v.PreProcess(self.data_input_file)
        self.prep.nan_check()
        self.correlation_matrix = self.prep.idle_sensors_check()

    def train(self, config_path=None):
        if config_path:
            self.config_path = config_path
            self.load_config()

        self.preprocessor()
        self.prep.time_interpolator(self.timestep)
        batch_size = int(np.round(self.batch_time / self.timestep, 0))
        dof_df = self.prep.convert_extract(self.dof, self.conversion)
        wve_df = self.prep.dataset["wave"]
        dofwve_df = pd.concat([dof_df, wve_df], axis=1).values

        # Normalize and scale
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self.scaler.fit_transform(dofwve_df)
        supervised_data = self.prep.series_to_supervised(
            scaled,
            wind_var_number=None,
            wave_var_number=len(self.dof) + 1,
            n_in=self.n,
            n_out=self.m,
            wind_predictor=self.wind_prediction,
            wave_predictor=self.wave_prediction)

        # Build, compile, and fit
        past_wind = future_wind = self.wind_prediction
        past_wave = future_wave = self.wave_prediction
        features = list(np.arange(1, len(self.dof) + 1, 1))
        labels = list(np.arange(1, len(self.dof) + 1, 1))

        self.mlstm_wrp.split_train_test(supervised_data=supervised_data,
                                        train_ratio=self.train_ratio,
                                        valid_ratio=self.valid_ratio,
                                        past_timesteps=self.n,
                                        future_timesteps=self.m,
                                        features=features,
                                        labels=labels,
                                        past_wind=past_wind,
                                        future_wind=future_wind,
                                        past_wave=past_wave,
                                        future_wave=future_wave)
        self.mlstm_wrp.build_and_compile_model(hidden_layer=self.hidden_layer,
                                               neuron_number=self.neuron_number,
                                               last_layer=len(labels),
                                               lr=self.lr,
                                               dropout=self.dropout)
        self.mlstm_wrp.model.fit(self.mlstm_wrp.train_X, self.mlstm_wrp.train_Y, epochs=self.epochs,
                                 batch_size=batch_size,
                                 validation_data=(self.mlstm_wrp.valid_X, self.mlstm_wrp.valid_Y), verbose=2,
                                 shuffle=False)

    def test(self):
        # y
        orig_Y = self.mlstm_wrp.test_Y
        dummy_array = np.zeros((orig_Y.shape[0], len(self.dof) + 1 if True else 0))
        dummy_array[:, :len(self.dof)] = orig_Y
        reversed_array = self.scaler.inverse_transform(dummy_array)
        y = reversed_array[:, :len(self.dof)]
        # yhat
        test_Y = self.mlstm_wrp.model.predict(self.mlstm_wrp.test_X)
        dummy_array = np.zeros((test_Y.shape[0], len(self.dof) + 1 if True else 0))
        dummy_array[:, :len(self.dof)] = test_Y
        reversed_array = self.scaler.inverse_transform(dummy_array)
        y_hat = reversed_array[:, :len(self.dof)]
        r_square = np.zeros(len(self.dof))
        mae = np.zeros(len(self.dof))
        labels = list(np.arange(1, len(self.dof) + 1, 1))
        for i, label in enumerate(labels):
            label_index = label - 1
            _, _, r_value_wrp, _, _ = linregress(y[:, label_index],
                                                 y_hat[:, label_index])
            mae[i] = mean_absolute_error(y[:, label_index], y_hat[:, label_index])
            r_square[i] = r_value_wrp ** 2
        return r_square, mae, y, y_hat

    def predict(self, time, state, wave, convert=True, history=0):
        """
        Predicts future states based on the past/present states and past/present/future wave data.
        Notes:
        1) All input does not have to be on the same timestamp as the trained network.
        2) State has a smaller length than wave since we are anticipating future wave readings.
        3) Name of the input columns do not have to be exactly as anticipated because this subroutines
         appropriately rename them. Length of state must, however, match with self.dof and time and wave has to have
         the same length.
        4) Make sure states if states are converted based on the config file [unit]s, toggle convert off. Otherwise,
        5) When history=0, this function only gives future prediction with no history of previous predictions.
        make sure you have the same units as the trained network before conversion.
        :param time: DataFrame containing corresponding time to wave
        :param state: DataFrame containing state variables up to the present time.
        :param wave: DataFrame containing wave data extending beyond the state data by m timesteps.
        :param convert: (default True) Flag to determine whether unit conversion is needed or not.
        :param history: (default 0) How far to the past (s) should the predictor provide data for. (must be positive)
        """
        input_timestep = time.iloc[-1] - time.iloc[-2]  # assuming timesteps do not change.
        # Step 0: Check if state has the correct number of columns
        if state.shape[1] != len(self.dof):
            raise ValueError(f"The 'state' DataFrame must have {len(self.dof)} columns, with DOF: {self.dof}.")

        # Ensure 'time' and 'wave' have the same length
        if len(time) != len(wave):
            raise ValueError("Time and wave data must have the same length.")

        # Step 1: Make state and wave the same length by appending synthetic future state data
        future_index_original = len(wave) - len(state)

        if future_index_original < 0:
            raise ValueError("State data should not exceed wave data in length.")
        synthetic_data = pd.DataFrame(0, index=np.arange(future_index_original), columns=state.columns)  # Zero-filled DataFrame
        state_updated = pd.concat([state, synthetic_data], ignore_index=True)

        # Step 2: Prepare data (bigdata -> smalldata -> preprocess -> interpolate -> concatenate
        bigdata = pd.concat([time, state_updated, wave], axis=1)
        red_idx = int((2*self.time_horizon + self.nm * self.time_horizon + history - 1*self.timestep)/input_timestep)
        if red_idx > bigdata.shape[0]:
            # Check if reducing historical prediction helps
            history -= (red_idx - bigdata.shape[0]) * input_timestep
            red_idx = int(
                (2 * self.time_horizon + self.nm * self.time_horizon + history - 1 * self.timestep) / input_timestep)
            if history < 0:
                raise ValueError("Not enough datapoints to produce prediction")
            else:
                print(f"setting history to {np.round(history, 2)}s")
        smalldata = bigdata.iloc[-red_idx:]
        # Rename columns
        smalldata.columns = ["Time"] + self.dof + ["wave"]

        data = p2v.PreProcess(raw_dataset=smalldata)
        data.time_interpolator(self.timestep)
        if convert:
            dof_df = data.convert_extract(self.dof, self.conversion)
        else:
            dof_df = data.dataset[self.dof]

        wve_df = data.dataset["wave"]
        dofwve_df = pd.concat([dof_df, wve_df], axis=1).values
        # Step 3: Scale based on the pre-assigned scaler
        scaled = self.scaler.transform(dofwve_df)

        # Step 4: Supervise data
        supervised_data = data.series_to_supervised(
            data=scaled,
            wind_var_number=None,
            wave_var_number=len(self.dof) + 1,
            n_in=self.n,
            n_out=self.m,
            wind_predictor=self.wind_prediction,
            wave_predictor=self.wave_prediction)

        # Step 5: Split all data as test data (train_ratio and valid_ratio are zero)
        self.mlstm_wrp.split_train_test(
            supervised_data=supervised_data,
            train_ratio=0.0,
            valid_ratio=0.0,
            past_timesteps=self.n,
            future_timesteps=self.m,
            features=list(np.arange(1, len(self.dof) + 1)),
            labels=list(np.arange(1, len(self.dof) + 1)),
            past_wind=self.wind_prediction,
            future_wind=self.wind_prediction,
            past_wave=self.wave_prediction,
            future_wave=self.wave_prediction)

        # Step 6: Predict using the model
        test_Y = self.mlstm_wrp.model.predict(self.mlstm_wrp.test_X)

        # Unscaling predicted data
        dummy_array = np.zeros((test_Y.shape[0], len(self.dof) + 1))  # Adjust the shape if necessary
        dummy_array[:, :len(self.dof)] = test_Y
        reversed_array = self.scaler.inverse_transform(dummy_array)
        t_hat = np.linspace(time.iloc[-(future_index_original + int(history/input_timestep))].item(),
                            time.iloc[-1].item(),
                            self.m + int(history/self.timestep))
        y_hat = pd.DataFrame(reversed_array[-(self.m + int(history/self.timestep)):, :len(self.dof)])
        t_pred = time[-(future_index_original + int(history/input_timestep)):].reset_index(drop=True)
        y_hat = pd.DataFrame(
            np.array([np.interp(t_pred, t_hat, y_hat[col]) for col in y_hat.columns]).T, columns=state.columns)

        # # Unify mean values
        # y_hat += dof_df.mean() - y_hat.mean()

        # shift by 1
        t_pred = t_pred.shift(1 * int(self.timestep / input_timestep))
        t_pred = t_pred.dropna()
        y_hat = y_hat.loc[t_pred.index].reset_index(drop=True)
        t_pred.reset_index(drop=True)
        return t_pred, y_hat