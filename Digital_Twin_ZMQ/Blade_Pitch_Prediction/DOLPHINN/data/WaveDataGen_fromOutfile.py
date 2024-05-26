import pandas as pd

# Read the original CSV file
file_path = '/Users/fredrikfleslandselheim/ROSCO/Digital_Twin_ZMQ/Outputs/Sim_Results/IEA15MW_FOCAL/custom_wind_wave_case/base/IEA15MW_FOCAL_0.out'
original_headers = pd.read_csv(file_path, sep='\s+', skiprows=6, nrows=0).columns.tolist()
original_data = pd.read_csv(file_path, sep='\s+', skiprows=list(range(6)) + [7], low_memory=False, names=original_headers, header=0)
original_data.rename(columns={'Wave1Elev': 'wave'}, inplace=True)
# Define the columns you want to keep
columns_to_keep = [
    'Time', 'wave']
TEST = "WaveData2305_Hs2Tp5_5"
# Save only the selected columns to a new CSV file
original_data[columns_to_keep].to_csv(f'/Users/fredrikfleslandselheim/ROSCO/Digital_Twin_ZMQ/Blade_Pitch_Prediction/DOLPHINN/data/{TEST}.csv', index=False)
