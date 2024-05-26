import os
import pandas as pd

# Read the original CSV file
this_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(this_dir,'Driver_Fatigue/Sim_Results/IEA15MW_FOCAL/custom_wind_wave_case/base/IEA15MW_FOCAL_0.out')
original_headers = pd.read_csv(file_path, sep='\s+', skiprows=6, nrows=0).columns.tolist()
original_data = pd.read_csv(file_path, sep='\s+', skiprows=list(range(6)) + [7], low_memory=False, names=original_headers, header=0)
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
original_data.rename(columns=rename_mapping, inplace=True)
columns_to_keep = ['Time', 'PtfmTDX', 'PtfmTDZ', 'PtfmTDY', 'PtfmRDX', 'PtfmRDY', 'PtfmRDZ', 'BlPitchCMeas', 'RotSpeed', 'wave']

TEST = "TrainingData_Hs2_75_Tp6"

output_path = os.path.join(this_dir, f'../Blade_Pitch_Prediction/DOLPHINN/data/{TEST}.csv')
original_data[columns_to_keep].to_csv(output_path, index=False)

print(f"Saved successfully to {output_path}")
