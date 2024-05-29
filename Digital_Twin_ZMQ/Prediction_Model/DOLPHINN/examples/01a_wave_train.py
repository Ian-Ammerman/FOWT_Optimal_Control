import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
from pathlib import Path
import yaml
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from vmod.dolphinn import DOLPHINN as DOL

# Configure
TEST = "TrainingData_Hs_2_75_Tp_6_testing"
CONFIG_FILE_PATH = "/Users/fredrikfleslandselheim/ROSCO/Digital_Twin_ZMQ/Blade_Pitch_Prediction/DOLPHINN/dol_input/training_param.yaml"
RESULTS_PATH = os.path.join("Digital_Twin_ZMQ", "Blade_Pitch_Prediction", "DOLPHINN", "training_results", f"{TEST}")
MODEL_PATH = os.path.join("Digital_Twin_ZMQ", "Blade_Pitch_Prediction", "DOLPHINN", "saved_models", f"{TEST}")

if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

# Call DOLPHINN
dol = DOL()
dol.train(config_path=CONFIG_FILE_PATH)
r_square, mae, y, y_hat = dol.test()

# Post-processing
fig = plt.figure(figsize=(12, 24))
gs = gridspec.GridSpec(len(dol.dof), 1)

for i, (label, unit) in enumerate(zip(dol.dof, dol.unit)):
    ax = plt.subplot(gs[i])
    ax.plot(y[:, i], label='Actual Blade Pitch Angle', color='black')
    ax.plot(y_hat[:, i], label='MLSTM Prediction', color='#3CB371', linestyle='--')
    ax.set_xlabel('t [s]')
    ax.set_ylabel(f"{label} {unit}")
    ax.set_xlim((0, 1250))
    ax.legend(loc='upper right')
        
    ax.grid()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "wave.pdf"), format="pdf")

# Save DOLPHINN model
dol.save(os.path.join(MODEL_PATH, "wave_model"))

# Save mean offset to config file
with open(CONFIG_FILE_PATH, 'r') as file:
    config = yaml.safe_load(file)

with open(CONFIG_FILE_PATH, 'w') as file:
    yaml.safe_dump(config, file)
