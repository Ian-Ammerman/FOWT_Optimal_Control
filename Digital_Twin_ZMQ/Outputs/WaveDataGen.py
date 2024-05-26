import os
import pandas as pd

# Read the original CSV file
this_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(this_dir,'Driver_Fatigue/Sim_Results/IEA15MW_FOCAL/custom_wind_wave_case/base/IEA15MW_FOCAL_0.out')
original_headers = pd.read_csv(file_path, sep='\s+', skiprows=6, nrows=0).columns.tolist()
original_data = pd.read_csv(file_path, sep='\s+', skiprows=list(range(6)) + [7], low_memory=False, names=original_headers, header=0)
original_data.rename(columns={'Wave1Elev': 'wave'}, inplace=True)
# Define the columns you want to keep
columns_to_keep = [
    'Time', 'wave']
TEST = "WaveData_Hs2_75_Tp6"

# Save only the selected columns to a new CSV file
output_path = os.path.join(this_dir, f'../Blade_Pitch_Prediction/DOLPHINN/data/{TEST}.csv')
original_data[columns_to_keep].to_csv(output_path, index=False)

# Print success message
print(f"Saved successfully to {output_path}")
