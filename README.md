# RUL Estimation and Predictive Control of Floating Offshore Wind Turbines

## Part 1: Fatigue model for RUL Estimation and Monitoring


## Part 2: MLSTM-WRP FOWT Blade Pitch Prediction Integration

This repository contains the implementation of a predictive control framework for a floating offshore wind turbine (FOWT) using a Multiplicative Long Short-Term Memory (MLSTM) neural network model. The primary goal is to predict the collective blade pitch angle in real-time, leveraging incoming wave elevation data and current FOWT measurements, to improve the response of the ROSCO blade pitch controller and reduce structural fatigue.

### Table of Contents
- [Introduction](#introduction)
- [Architecture](#architecture)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
- [Contributing](#contributing)
- [License](#license)

### Introduction

This project integrates an MLSTM model with OpenFAST and ROSCO to predict the future response of a FOWT. The prediction model aims to provide a time advantage to blade actuators by predicting future states based on wave elevation data and current FOWT measurements.

The primary objective is to set the framework for future applications for implementing MLSTM-prediction in the ROSCO controller during simulation, using Yuksel R. Alkarem's predictive model. For this specific example, the aim is to reduce wave-induced motions, thereby decreasing structural fatigue and increasing the remaining useful life (RUL) of the FOWT by predicting future collective blade pitch angle based on incoming wave data, and sending setpoints to the ROSCO controller. 

### Blade Pitch Prediction Architecture

The MLSTM-WRP model is integrated with OpenFAST and ROSCO through a series of scripts and configurations:

- `Driver.py`: Main script to configure and run the OpenFAST simulation with the prediction model.
- `Blade_Pitch_Prediction`: Folder containing prediction model scripts, as listed below.
- `pitch_prediction.py`: Contains the logic for accumulating data batches and interfacing with the MLSTM model.
- `prediction_functions.py`: Utility functions such as buffering\saturating prediction offsets, plotting and saving data. 
- `wave_predict.py`: Script developed in collaboration with Yuksel R. Alkarem for wave prediction.
- `DOLPHINN`: An MLSTM framework developed by Yuksel R. Alkarem for predicting FOWT behavior based on incoming wave data.

### Measurement states used from ROSCO for training and prediction:

- 'BlPitchCMease':
- 'PtfmTDX': Surge
- 'PtfmTDZ': Heave
- 'PtfmTDY': Sway
- 'PtfmRDX': Roll
- 'PtfmRDY': Pitch
- 'PtfmRDZ': Roll
  

### Prediction Model Usage with OpenFAST:
1. Configure the `Driver.py` script with the desired load case, model, and simulation settings.

Prediction model configuration during simulation in the `wfc_controller` in `Driver.py`:

- Specify which FOWT state to use for prediction and monitoring. Choosing "BlPitchCMeas" allows for Buffer and Saturate-initiation, while other DOFS only provide prediction and monitoring:
    ```python
    FOWT_pred_state = 'BlPitchCMeas'
     ``` 
- Specify trained MLSTM-model:
    ```python
    MLSTM_MODEL_NAME = TrainingData_Hs_2_75_Tp_6 # Trained MLSTM model
     ```

- Make sure that the `WAVE_DATA_FILE` is a csv-timeseries, matching the sea state specified in Load Case:
    ```python
    WAVE_DATA_FILE = WaveData_LC2 # Example wave file for LC2
     ```
- To send offset between blade pitch MLSTM-prediction and actual blade pitch angle:
    ```python
    Prediction = True 
     ```
    
-  For real time prediction plotting:
    ```python
    plot_figure = True
     ```

- To saturate offset between prediction and actual measurement:
    ```python
    Pred_Saturation = True
     ```

- If observing an amplitude offset between prediction and measurement, an error may be defined to correct the predictions:
    ```python
    pred_error = 1.4 # [deg]
     ```
    
2. To run a simulation with Load Case 2, the `Driver.py` script should be configured as follows in the `__init__`:
  ```python
  self.Load_Case = 2
   ```

3. Run the main driver script:
```bash
python Driver.py
```

#### MLSTM model training:

In order to train a custom MLSTM-model, this is done by running the following script:

`/ROSCO/Digital_Twin_ZMQ/Blade_Pitch_Prediction/DOLPHINN/examples/01a_wave_train.py`

Specify training data and parameters in:

`/ROSCO/Digital_Twin_ZMQ/Blade_Pitch_Prediction/DOLPHINN/dol_input/training_param.yaml`

