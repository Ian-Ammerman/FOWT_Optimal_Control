import pandas as pd

# Path to the original wind file
original_file = '/Users/fredrikfleslandselheim/ROSCO/project/Test_Cases/IEA-15-240-RWT-UMaineSemi/Wind_Files/W02_fullScale_R02_20230606.wnd'
# Path for the modified wind file
modified_file = '/Users/fredrikfleslandselheim/ROSCO/project/Test_Cases/IEA-15-240-RWT-UMaineSemi/Wind_Files/W02_fullScale_R02_20230606_modified.wnd'

# Read the wind file, skipping the headers
wind_data = pd.read_csv(original_file, sep='\s+', skiprows=3, header=None,
                        names=['Time', 'WindSpeed', 'WindDir', 'VertSpeed', 'HorizShear', 'VertShear', 'LinVShear', 'Gust'])

# Adjust the time column to start at 0.0
wind_data['Time'] -= wind_data['Time'].iloc[0]
wind_data = wind_data.applymap(lambda x: f'{x:.6f}' if isinstance(x, float) else x)

# Then write the modified data to a new file, including the original headers

# Write the modified data to a new file, including the original headers
with open(modified_file, 'w') as f:
    f.write('! Wind file created 06-Jun-2023 12:39:53\n')
    f.write('! Time\tWind\tWind\tVert.\tHoriz.\tVert.\tLinV\tGust\n')
    f.write('! \tSpeed\tDir \tSpeed\tShear\tShear\tShear\tSpeed\n')
wind_data.to_csv(modified_file, sep='\t', mode='a', index=False, header=False)

print(f'Modified wind file saved as: {modified_file}')
