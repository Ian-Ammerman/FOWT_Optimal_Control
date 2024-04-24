from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, save_model, load_model
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from scipy.stats import linregress
from vmod import get_psd
from sklearn.preprocessing import MinMaxScaler
import joblib
from scipy.signal import butter, filtfilt

class PreProcess():
    def __init__(self, data_input_file=None, raw_dataset=None):
        self.scaler = None
        self.dynHs = None
        self.dynTp = None
        self.dynTi = None
        if not data_input_file:
            self.raw_dataset = raw_dataset
        elif not raw_dataset:
            self.raw_dataset = pd.read_csv(data_input_file)

        self.dataset = self.raw_dataset.copy()
        self.timestep = (self.raw_dataset['Time'].iloc[1] - self.raw_dataset['Time'].iloc[0])
        self.sampling_rate = 1/self.timestep

    def subtract_mean(self):
        mean_values = self.dataset.mean()
        mean_values['Time'] = 0  # keep time as is
        self.dataset = self.dataset.subtract(mean_values, axis=1)

    def nan_check(self):
        print(f"Checking if there is any NaN values in the raw dataset")
        pd.DataFrame(self.dataset.isna().sum(), columns=['number of NaN in the raw dataset'])
        self.dataset = self.dataset.dropna()

    def filter(self, direction, freq_cutoff):
        """
        Apply a low-pass Butterworth filter to the dataset.
        :param direction: 'low pass' for low-pass filtering.
        :param freq_cutoff: The cutoff frequency for the filter.
        """
        # Calculate the Nyquist frequency
        nyquist_freq = 0.5 * self.sampling_rate

        # Calculate the cutoff frequency as a fraction of the Nyquist frequency
        normalized_cutoff_freq = freq_cutoff / nyquist_freq

        # Design the Butterworth filter
        b, a = butter(N=6, Wn=normalized_cutoff_freq, btype=direction, analog=False)

        # Apply the filter to each column except 'Time'
        for column in self.dataset.columns:
            if column != 'Time':
                self.dataset[column] = filtfilt(b, a, self.dataset[column])

    def idle_sensors_check(self):
        print(f"Checking if there is any idle sensors")
        correlation_matrix = self.dataset.corr()
        idle_sensors = np.array(correlation_matrix[correlation_matrix.isna().sum() == self.dataset.shape[1]].index)
        if not len(idle_sensors) == 0:
            print(f"The following sensors are idle: {idle_sensors}")
        self.dataset = self.dataset.drop(columns=idle_sensors)
        return self.dataset.corr()

    def time_interpolator(self, timestep):
        """
        interpolate data based on a given timestep
        :param timestep: the timestep requested
        :return: a new interpolated dataset
        """
        new_time_range = np.arange(self.dataset['Time'].min(), self.dataset['Time'].max(), timestep)
        dataset_interpolated = pd.DataFrame(new_time_range, columns=['Time'])
        for col in self.dataset.columns:
            if col != 'Time':  # Skip the 'Time' column
                dataset_interpolated[col] = np.interp(new_time_range, self.dataset['Time'], self.dataset[col])

        self.dataset = dataset_interpolated
        self.timestep = timestep

    def convert_extract(self, dof, conversion):
        df = self.dataset[dof]
        df = df.mul(conversion, axis=1)
        return df

    def series_to_supervised(self, data, wind_var_number, wave_var_number, n_in=1, n_out=1,
                             dropnan=True, wind_predictor=False, wave_predictor=False):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            wind_var_number: the index of the wind column in the data
            wave_var_number: the index of the wave column in the data
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()

        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%.2f)' % (j + 1, i)) for j in range(n_vars)]

        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            names += [(f'var%d(t+%.2f)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)

        if wind_var_number:
            # WIND: replace varx with wind
            new_columns = [col.replace(f"var{wind_var_number}", "wind") for col in agg.columns]
            agg.columns = new_columns
        if wave_var_number:
            # WAVE: replace varx with wave
            new_columns = [col.replace(f"var{wave_var_number}", "wave") for col in agg.columns]
            agg.columns = new_columns

        # Interpolate wind future data if needed
        if wind_predictor:
            # Define interpolation points
            x = np.linspace(0, n_out - 1, n_in)
            wind_input_name = [f"wind(t+{xi:.2f})" for xi in x]

            # Extract relevant columns for interpolation
            wind_cols = [f'wind(t+{j:.2f})' for j in range(0, n_out)]
            wind_data = agg[wind_cols].to_numpy()

            # Perform vectorized interpolation
            xp = np.arange(0, n_out)
            wind_interpolated = np.array([np.interp(x, xp, wind_row) for wind_row in wind_data])

            # Create DataFrame from interpolated data
            wind_df = pd.DataFrame(wind_interpolated, columns=wind_input_name)

            # Concatenate the new DataFrame with the existing one
            agg = agg.drop(columns=wind_cols)

            # The old way to concatenate
            agg = pd.concat([agg, wind_df], axis=1)

            # The new way to concatenate
            # agg = pd.concat([agg.reset_index(drop=True), wind_df.reset_index(drop=True)], axis=1)

        # Interpolate wave future data if needed
        if wave_predictor:
            # Define interpolation points
            x = np.linspace(0, n_out - 1, n_in)
            wave_input_name = [f"wave(t+{xi:.2f})" for xi in x]

            # Extract relevant columns for interpolation
            wave_cols = [f'wave(t+{j:.2f})' for j in range(0, n_out)]
            wave_data = agg[wave_cols].to_numpy()

            # Perform vectorized interpolation
            xp = np.arange(0, n_out)
            wave_interpolated = np.array([np.interp(x, xp, wave_row) for wave_row in wave_data])

            # Create DataFrame from interpolated data
            wave_df = pd.DataFrame(wave_interpolated, columns=wave_input_name)

            # Concatenate the new DataFrame with the existing one
            agg = agg.drop(columns=wave_cols)

            agg = pd.concat([agg.reset_index(drop=True), wave_df.reset_index(drop=True)], axis=1)

        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)

        if wind_var_number:
            # WIND: replace varx with wind
            new_columns = [col.replace(f"var{wind_var_number}", "wind") for col in agg.columns]
            agg.columns = new_columns
        if wave_var_number:
            # WAVE: replace varx with wave
            new_columns = [col.replace(f"var{wave_var_number}", "wave") for col in agg.columns]
            agg.columns = new_columns

        return agg

    def zero_crossing(self, time, signal):
        # Initialize variables
        stime = []
        sfinder = []
        T = []
        H = []
        # Finding zero up crossings
        for t in range(1, len(signal)):
            if signal[t - 1] < 0 and signal[t] > 0:
                stime.append(time[t])
                sfinder.append(np.where(time[t] == time)[0][0])

        # Calculating T and H
        for i in range(1, len(sfinder)):
            T.append(stime[i] - stime[i - 1])
            H.append(np.ptp(signal[sfinder[i - 1]:sfinder[i]]))

        return np.array(T), np.array(H), np.array(stime), np.array(sfinder)

    def average_top_third(self, signal):
        # Sort H in descending order
        signal_sorted = np.sort(signal)[::-1]

        # Calculate the length of the top one-third
        top_third_length = int(np.ceil(len(signal) / 3))

        # Select the top one-third elements
        top_third = signal_sorted[:top_third_length]

        # Calculate the average
        average = np.mean(top_third)

        return average

    def dynamic_sig_char(self, chkpnt_no, significant_length):
        significant_length_idx = int(significant_length / self.dataset['Time'].iloc[1])
        self.dynHs = np.zeros(chkpnt_no)
        self.dynTp = np.zeros(chkpnt_no)
        self.dynTi = np.zeros(chkpnt_no)
        chkpnts = [int(num) for num in np.linspace(significant_length_idx, len(self.dataset) - 1, chkpnt_no)]
        for i, chkpnt in enumerate(chkpnts):
            T, H, _, _ = self.zero_crossing(np.array(
                self.dataset['Time'].iloc[chkpnt - significant_length_idx:chkpnt]),
                np.array(self.dataset['waveStaff5'].iloc[chkpnt - significant_length_idx:chkpnt]
                         - np.mean(self.dataset['waveStaff5'].iloc[chkpnt - significant_length_idx:chkpnt])))
            self.dynHs[i] = self.average_top_third(H)
            self.dynTp[i] = self.average_top_third(T) * 1.05
            self.dynTi[i] = self.dataset['Time'].iloc[chkpnt]


class MLSTM:
    def __init__(self):
        self.scaler = None
        self.test_X = None
        self.valid_X = None
        self.train_X = None
        self.test_Y = None
        self.valid_Y = None
        self.train_Y = None

        self.model = Sequential()

    def fit_scaler(self, data):
        self.scaler.fit(data)

    def split_train_test(self, supervised_data, train_ratio, valid_ratio, past_timesteps, future_timesteps,
                         features, labels, past_wind=False, future_wind=False, past_wave=False, future_wave=False):
        """
        :param supervised_data:
        :param train_ratio: the
        ratio of the supervised dataset to be used for training
        :param valid_ratio: the ratio of the supervised
        dataset to be used for validation
        :param past_timesteps: the value (n) of timesteps of observations in the past
        :param future_timesteps: the value (n) of timesteps of observations to be extracted in the future
        :param features: [list] the `index` of variable(s) that are used as input.
        :param labels: [list] the 'index' of variable(s) to be forecasted as output.
        :param past_wind: past wave elevation data as input
        :param future_wind: activates the
        ability to allow predicted wind in the futures to be input as well (if there is wind, the past version of it
        will be used)
        :param past_wave: past wave elevation data as input
        :param future_wave: activates the
        ability to allow predicted waves in the futures to be input as well (if there is wave, the past version of it
        will be used)
        :return: nothing. The class updates itself with the train, valid, and test datasets
        """
        # split into input and outputs
        # Convert var_numbers to list if it's a single number
        if not isinstance(features, list):
            var_numbers = [features]

        if not isinstance(labels, list):
            var_numbers = [labels]

        input_columns = self.extract_input_columns(supervised_data.columns, features, past_timesteps,
                                                   past_wind, future_wind, past_wave, future_wave)
        num_features = len(features) + (1 if past_wind else 0) + (1 if future_wind else 0) + \
                       (1 if past_wave else 0) + (1 if future_wave else 0)
        output_columns = self.extract_output_columns(supervised_data.columns, labels, future_timesteps)

        # Selecting the columns from the dataframe
        input_super_data = supervised_data[input_columns]
        output_super_data = supervised_data[output_columns]

        # split into train and test sets
        split_idx_train = int(len(input_super_data) * train_ratio)
        split_idx_valid = int(len(input_super_data) * valid_ratio)

        # Train
        train_X = input_super_data.values[:split_idx_train, :]
        valid_X = input_super_data.values[split_idx_train:split_idx_train + split_idx_valid, :]
        test_X = input_super_data.values[split_idx_train + split_idx_valid:, :]

        train_Y = output_super_data.values[:split_idx_train, :]
        valid_Y = output_super_data.values[split_idx_train:split_idx_train + split_idx_valid, :]
        test_Y = output_super_data.values[split_idx_train + split_idx_valid:, :]

        # reshape input to be 3D [samples, timesteps, features]
        self.train_X = train_X.reshape((train_X.shape[0], past_timesteps, num_features))
        self.valid_X = valid_X.reshape((valid_X.shape[0], past_timesteps, num_features))
        self.test_X = test_X.reshape((test_X.shape[0], past_timesteps, num_features))

        # self.train_Y = train_Y.reshape(train_Y.shape[0], future_timesteps, len(labels))
        # self.valid_Y = valid_Y.reshape(valid_Y.shape[0], future_timesteps, len(labels))
        # self.test_Y = test_Y.reshape(test_Y.shape[0], future_timesteps, len(labels))

        self.train_Y = train_Y
        self.valid_Y = valid_Y
        self.test_Y = test_Y

    def extract_input_columns(self, columns, features, past_timesteps, past_wind, future_wind, past_wave, future_wave):
        # Extracting input columns:
        # Lists for different types of columns
        var_columns = [col for col in columns if any(f'var{var_num}(t-' in col for var_num in features)]
        past_wind_columns = [col for col in columns if 'wind(t-' in col] if past_wind else []
        future_wind_columns = [col for col in columns if
                              'wind(t+' in col or 'wind(t)' in col] if future_wind else []
        past_wave_columns = [col for col in columns if 'wave(t-' in col] if past_wave else []
        future_wave_columns = [col for col in columns if
                              'wave(t+' in col or 'wave(t)' in col] if future_wave else []

        num_features = len(features) + (1 if past_wind else 0) + (1 if future_wind else 0) + \
                       (1 if past_wave else 0) + (1 if future_wave else 0)

        # Interleaving columns
        input_columns = []
        for i in range(past_timesteps):
            for j in range(num_features):
                if j < len(features):
                    input_columns.append(var_columns[len(features) * i + j])
                else:
                    if past_wind:
                        input_columns.append(past_wind_columns[i])
                    if future_wind:
                        input_columns.append(future_wind_columns[i])
                    if past_wave:
                        input_columns.append(past_wave_columns[i])
                    if future_wave:
                        input_columns.append(future_wave_columns[i])
                    break
        return input_columns

    def extract_output_columns(self, columns, labels, future_timesteps):
        output_columns = [col for col in columns if
                          any(f'var{var_num}(t+{float(future_timesteps - 1):.2f})' in col for var_num in labels)]
        return output_columns

    def build_and_compile_model(self, hidden_layer, neuron_number, last_layer, lr=0.001, dropout=0.0):
        neuron_per_layer = int(np.round(neuron_number / hidden_layer, 0))
        self.model.add(LSTM(neuron_per_layer, input_shape=(self.train_X.shape[1], self.train_X.shape[2]),
                            return_sequences=True if hidden_layer > 1 else False, dropout=dropout))

        # Add additional LSTM layers
        for _ in range(1, hidden_layer):  # Start the range from 1 since we already added the first LSTM layer
            # Only the final LSTM layer should not return sequences, hence check if it's the last one
            return_sequences = _ < hidden_layer - 1
            self.model.add(LSTM(neuron_per_layer, return_sequences=return_sequences, dropout=dropout))

        # Add the output Dense layer
        self.model.add(Dense(last_layer))
        optimizer = Adam(learning_rate=lr)
        self.model.compile(optimizer=optimizer, loss='mae')
        self.model.summary()

    def fit_model(self, epochs, batch_size):
        self.model.fit(self.train_X, self.train_Y,
                       validation_data=(self.valid_X, self.valid_Y),
                       epochs=epochs,
                       batch_size=batch_size,
                       verbose=2,
                       shuffle=False)

    def run_rt(self, batch):
        """
        batch: DO NOT SCALE BATCH WHEN CALLING run_rt
        """

    def save_model(self, model_dir, scaler_dir):
        save_model(self.model, model_dir)
        joblib.dump(self.scaler, scaler_dir)

    def load_model(self, model_dir, scaler_dir):
        self.model = load_model(model_dir)
        self.scaler = joblib.load(scaler_dir)


class PostProcess():
    def __init__(self, scaler, Y_physical, Y_virtual, timestep):
        self.virtual_psd = None
        self.physical_psd = None
        self.f = None
        self.mean_physical = None
        self.t = np.arange(0, timestep * len(Y_virtual), timestep)
        self.dof_no = Y_physical.shape[1]
        self.virtual = self.wrp_data_unscale(scaler, Y_physical)
        self.physical = self.wrp_data_unscale(scaler, Y_virtual)
        self.unite_mean_value()

    def wrp_data_unscale(self, scaler, Y):
        num_original_features = len(scaler.data_max_)
        dummy_array = np.zeros((Y.shape[0], num_original_features))
        dummy_array[:, :self.dof_no] = Y
        reversed_array = scaler.inverse_transform(dummy_array)
        unscaled = reversed_array[:, :self.dof_no]
        return unscaled

    def unite_mean_value(self):
        self.mean_physical = self.physical.mean(axis=0)
        mean_virtual = self.virtual.mean(axis=0)
        self.virtual += self.mean_physical - mean_virtual

    def go_freq(self, s, f_low, f_high):
        self.f, _ = get_psd.get_PSD_limited(
            self.t,
            self.physical[:, 0] - self.mean_physical[0],
            s,
            f_low,
            f_high)
        self.physical_psd = np.zeros((len(self.f), self.dof_no))
        self.virtual_psd = np.zeros((len(self.f), self.dof_no))
        for dof in range(self.dof_no):
            self.f, self.physical_psd[:, dof] = get_psd.get_PSD_limited(
                self.t,
                self.physical[:, dof] - self.mean_physical[dof],
                s,
                f_low,
                f_high)
            _, self.virtual_psd[:, dof] = get_psd.get_PSD_limited(
                self.t,
                self.virtual[:, dof] - self.mean_physical[dof],
                s,
                f_low,
                f_high)

    def find_R2(self):
        R2 = []
        for label in range(self.dof_no):
            _, _, r_value_wrp, _, _ = linregress(self.physical[:, label],
                                                 self.virtual[:, label])
            R2.append(r_value_wrp)
        return R2

    def find_R2_freq(self):
        R2 = []
        for label in range(self.physical.shape[1]):
            _, _, r_value_wrp, _, _ = linregress(self.physical_psd[:, label],
                                                 self.virtual_psd[:, label])
            R2.append(r_value_wrp)
        return R2