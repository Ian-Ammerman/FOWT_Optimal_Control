import subprocess
import time

# Start blade_pitch_controller.py
blade_proc = subprocess.Popen(['python', '/home/hpsauce/ROSCO/project/blade_pitch_controller.py'])
print("Started blade_pitch_controller.py")

time.sleep(5)  # Adjust the sleep time as necessary

# Start pitch_prediction.py
pitch_proc = subprocess.Popen(['python', '/home/hpsauce/ROSCO/project/Prediction/pitch_prediction.py'])
print("Started pitch_prediction.py, waiting for it to initialize...")



# Wait for the scripts to complete
pitch_proc.wait()
blade_proc.wait()