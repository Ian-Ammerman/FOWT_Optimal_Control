import pandas as pd

# Define the file paths and columns to keep
file_path = '/Users/fredrikfleslandselheim/ROSCO/Digital_Twin_ZMQ/Blade_Pitch_Prediction/DOLPHINN/data/TrainingDataCase3.csv'
columns_to_keep = ['Time', 'wave']
TEST = "WaveData_Hs_3_5_Tp_6_5"

# Construct the output file path using the TEST variable
output_file = f'/Users/fredrikfleslandselheim/ROSCO/Digital_Twin_ZMQ/Blade_Pitch_Prediction/DOLPHINN/data/{TEST}.csv'

# Read the original CSV file
original_data = pd.read_csv(file_path, sep=',', low_memory=False)

# Save only the selected columns to a new CSV file
original_data[columns_to_keep].to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}")
