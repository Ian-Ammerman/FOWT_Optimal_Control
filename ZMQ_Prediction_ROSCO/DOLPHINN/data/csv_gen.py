import numpy as np
import pandas as pd
# Setting the new number of rows and timestep
num_rows_1000 = 100
timestep_001 = 0.5
columns = ['Time', 'wave', 'PtfmTDX', 'PtfmTDY', 'PtfmTDZ', 'PtfmRDX', 'PtfmRDY', 'PtfmRDZ', 'FA_Acc']


# Generating new random data between 1 and 5
data_1000 = np.random.uniform(1, 5, size=(num_rows_1000, len(columns) - 1))
df_1000 = pd.DataFrame(data_1000, columns=columns[1:])

# Adding the 'Time' column with a timestep of 0.01
df_1000['Time'] = np.arange(timestep_001, num_rows_1000 * timestep_001 + timestep_001, timestep_001)

# Rearrange columns to place 'Time' at the beginning and round to 2 decimals
df_1000 = df_1000[columns].round(4)

# Saving the updated DataFrame to a new CSV file
csv_1000_path = "/home/hpsauce/ROSCO/ZMQ_Prediction_ROSCO/DOLPHINN/data/100r_0_5.csv"
df_1000.to_csv(csv_1000_path, index=False)

csv_1000_path
