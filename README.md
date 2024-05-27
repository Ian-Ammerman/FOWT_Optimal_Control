# MLSTM-WRP FOWT Blade Pitch Prediction Integration

This repository contains the implementation of a predictive control framework for a floating offshore wind turbine (FOWT) using a Multiplicative Long Short-Term Memory (MLSTM) neural network model. The primary goal is to predict the collective blade pitch angle in real-time, leveraging incoming wave elevation data and current FOWT measurements, to improve the response of the ROSCO blade pitch controller and reduce structural fatigue.

## Table of Contents
- [Introduction](#introduction)
- [Architecture](#architecture)
- [Usage](#usage)
- [Scripts Overview](#scripts-overview)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project integrates an MLSTM model with OpenFAST and ROSCO to predict the future response of a FOWT. The prediction model aims to provide a time advantage to blade actuators by predicting future states based on wave elevation data and current FOWT measurements.

The primary objective is to set the framework for future applications for implementing MLSTM-prediction in the ROSCO controller during simulation, using Yuksel R. Alkarem's predictive model. For this specific example, the aim is to reduce wave-induced motions, thereby decreasing structural fatigue and increasing the remaining useful life (RUL) of the FOWT by predicting future collective blade pitch angle based on incoming wave data, and sending setpoints to the ROSCO controller. 

## Architecture

The MLSTM-WRP model is integrated with OpenFAST and ROSCO through a series of scripts and configurations:
- `Driver.py`: Main script to configure and run the OpenFAST simulation with the prediction model.
- `pitch_prediction.py`: Contains the logic for accumulating data batches and interfacing with the MLSTM model.
- `prediction_functions.py`: Utility functions for buffering and saturating prediction offsets.
- `wave_predict.py`: Script developed in collaboration with Yuksel R. Alkarem for wave prediction.
- `DOLPHINN`: An MLSTM framework developed by Yuksel R. Alkarem for predicting FOWT behavior based on incoming wave data.
## Usage

1. Configure the `Driver.py` script with the desired load case, model, and simulation settings.
2. Run the main driver script:
    ```bash
    python Driver.py
    ```

### Example
To run a simulation with Load Case 2, the `Driver.py` script should be configured as follows:
```python
self.Load_Case = 2
