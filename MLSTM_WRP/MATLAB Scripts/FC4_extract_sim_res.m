%% Created by Rudy Alkarem
% This script extracts the simulations results for wave only, wind only,
% and wind and wave tests and saves them in an excel file
clear;close all;clc;
%% Extracting data
% Wave only data
load("../../Simulations/WaveOnly_FOCAL_C4/" + ...
    "FOCAL_4.1_M02D00T00_AC01_E22W00_R01_Z1_A1.mat")

T = array2table(channels, 'VariableNames', labels);
csv_name = '../data/waveonly.csv';
writetable(T, csv_name);

% Wind only data
load("../../Simulations/WindOnly_FOCAL_C4/" + ...
    "FOCAL_4.2_M02D01T01_AC01_E00W02_R01_Z1_A2.mat")

T = array2table(channels, 'VariableNames', labels);
csv_name = '../data/windonly.csv';
writetable(T, csv_name);

% Wind + Wave data
load("../../Simulations/WindWave_Rated_FOCAL_C4/" + ...
    "FOCAL_4.2_M02D01T01_AC01_E30W02_R01_Z1_A2.mat")

T = array2table(channels, 'VariableNames', labels);
csv_name = '../Data/windwave.csv';
writetable(T, csv_name);

% Extract wind data
filePath = "../../Wind_Files/W02_fullScale_R02_20230606.wnd";
opts = detectImportOptions(filePath, 'FileType', 'text');
windData = readtable(filePath, opts);
T = array2table([windData.x_, windData.Speed]);
csv_name = '../Data/W02_winddata.csv';
writetable(T, csv_name);
