import os
import time
import pandas as pd
from io import StringIO

class DataCollect:
    def __init__(self, input_dir, chunk_duration):
        self.input_dir = input_dir
        self.output_dir = os.path.join(input_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_path = os.path.join(input_dir, "Sim_Results", "IEA15MW_FOCAL", "custom_wind_wave_case", "base", "IEA15MW_FOCAL_0.out")
        self.columns_to_keep = ['Time', 'RootFzb1', 'RootFzb2', 'RootFzb3', 'RootMxb1', 'RootMxb2', 'RootMxb3', 'RootMyb1', 'RootMyb2', 'RootMyb3', 'TwrBsFzt', 'TwrBsMxt', 'TwrBsMyt']
        self.last_read_position = 0
        self.data_list = []
        self.data_frame = pd.DataFrame(columns=self.columns_to_keep)
        self.chunk_duration = chunk_duration
        print(f"DataCollect initialized with directory: {self.input_dir} and chunk duration: {self.chunk_duration}")

    def file_is_valid(self):
        is_valid = os.path.isfile(self.file_path) and os.path.getsize(self.file_path) > 0
        #print(f"File valid status: {is_valid}")
        return is_valid

    def read_and_filter_data(self):
        # print("Attempting to read data...")
        if not self.file_is_valid():
            print(f"File is not valid or not ready for reading: {self.file_path}")
            return pd.DataFrame()

        try:
            with open(self.file_path, 'r') as file:
                file.seek(self.last_read_position)
                chunk_data = StringIO()
                for i, line in enumerate(file):
                    if self.last_read_position == 0 and i < 8:  # Skip header and unit lines
                        if i in list(range(6)) + [7]:
                            continue
                    chunk_data.write(line)
                self.last_read_position = file.tell()

            chunk_data.seek(0)  # Reset position for reading
            # Example indices for usecols based on an assumption of column placement
            usecols_indices = [0, 38, 39, 40, 41, 42, 43, 47, 48, 49, 86, 87, 88] 
            data = pd.read_csv(chunk_data, sep='\s+', header=None, usecols=usecols_indices, names=self.columns_to_keep, skiprows=1)
            if not data.empty:
                self.data_list.extend(data.to_dict('records'))
                #print(f"Read {len(data)} new rows, total accumulated: {len(self.data_list)}")
            return data
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return pd.DataFrame()

    def process_data(self):
        if self.data_list:
            new_df = pd.DataFrame(self.data_list)
            self.data_frame = pd.concat([self.data_frame, new_df], ignore_index=True)
            self.data_list = []  # Clear the list after processing
            #print(f"Processed {len(new_df)} new rows. Total rows in DataFrame: {len(self.data_frame)}")
            return new_df
        return pd.DataFrame(columns=self.columns_to_keep)

if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(this_dir, "../Outputs")
    generator = DataCollect(input_dir, chunk_duration=100)
    new_data = generator.read_and_filter_data()
    if not new_data.empty:
        print("Data processed for current time.")
