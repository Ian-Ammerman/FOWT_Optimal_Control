import pandas as pd

# Read the original CSV file
file_path = '/Users/fredrikfleslandselheim/ROSCO/Digital_Twin_ZMQ/Outputs/Sim_Results/IEA15MW_FOCAL/custom_wind_wave_case/base/IEA15MW_FOCAL_0.out'

# Read the headers first
original_headers = pd.read_csv(file_path, sep='\s+', skiprows=6, nrows=0).columns.tolist()

# Read the data with original headers and skip the specified rows
original_data = pd.read_csv(file_path, sep='\s+', skiprows=list(range(6)) + [7], low_memory=False, names=original_headers, header=0)

# Define the mapping of old column names to new column names
rename_mapping = {
    'PtfmHeave': 'PtfmTDZ',
    'PtfmPitch': 'PtfmRDX',
    'PtfmRoll': 'PtfmRDY',
    'PtfmSurge': 'PtfmTDX',
    'PtfmSway': 'PtfmTDY',
    'PtfmYaw': 'PtfmRDZ',
    'BldPitch1': 'BlPitchCMeas',
    'RotSpeed': 'RotSpeed',  # remains unchanged
    'Wave1Elev': 'wave'
}

# Rename the columns
original_data.rename(columns=rename_mapping, inplace=True)

# Define the columns you want to keep
columns_to_keep = ['Time', 'PtfmTDX', 'PtfmTDZ', 'PtfmTDY', 'PtfmRDX', 'PtfmRDY', 'PtfmRDZ', 'BlPitchCMeas', 'RotSpeed', 'wave']

# Save only the selected columns to a new CSV file
original_data[columns_to_keep].to_csv('Digital_Twin_ZMQ/Blade_Pitch_Prediction/DOLPHINN/data/TrainingData_Hs_2_75_Tp_6.csv', index=False)
