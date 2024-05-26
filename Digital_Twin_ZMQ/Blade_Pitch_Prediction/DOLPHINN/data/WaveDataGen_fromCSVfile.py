import os
import pandas as pd

# Read the original CSV file
this_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(this_dir,'TrainingData_Hs2_75_Tp6.csv')
original_data = pd.read_csv(file_path, sep=',', low_memory=False)
columns_to_keep = ['Time', 'wave']

TEST = "WaveData_Hs_2_75_Tp_6"

# Construct the output file path using the TEST variable
output_path = os.path.join(this_dir, f'{TEST}.csv')
original_data[columns_to_keep].to_csv(output_path, index=False)

# Print success message
print(f"Saved successfully to {output_path}")
