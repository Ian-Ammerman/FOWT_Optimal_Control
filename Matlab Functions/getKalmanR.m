function [R,Plant,dt,time,wave_input] = getKalmanR(test_dir,measurements,y_op,low_cutoff)

% Written By: Ian Ammerman
% Last Modified: 12/18/23
%
% [R,Plant] = getKalmanR computes the measurement covariance for the real
% system from test data located at test_dir. Outputs covariance matrix R
% and plant measurements as matrix.
%
% Inputs:
% ---------
% test_dir - global path to test files
% measurements - desired measurements, in order, as a cell array
% y_op - measurement operating point, as vector
% low_cutoff - cutoff frequency for low-pass filter

%% Load in test results
test_file = sprintf('%s\\Test_Results.mat',test_dir);
load(test_file,'test_results');

time = test_results.Time;
Fs = length(time)/max(time); % Sampling frequency
dt = max(time)/length(time); % Time step
wave_input = test_results.Wave1Elev;

%% Initialize matrix of measurements
[Plant,lowFiltered,highFiltered] = deal(zeros(length(time),length(measurements)));
R = zeros(size(measurements));

%% Low-pass filter cutoff frequency
Ffilter = low_cutoff;

%% Load measurements into matrix for filtering
for i = 1:length(measurements)

    % Load in measurements
    try
        if strcmp(measurements{i},'TwrBsMyt')
            vals = test_results.(measurements{i})*10^-3;
        else
            vals = test_results.(measurements{i});
        end
    catch
        error(sprintf('Could not load %s from test results.',measurements{i}));
    end

    % Remove operating point from data
    vals = vals - y_op(i);

    % Low-pass filter the data
    lowFiltered(:,i) = lowpass(vals,Ffilter,Fs);
    Plant(:,i) = lowFiltered(:,i);

    % High-pass filter the data
    highFiltered(:,i) = highpass(vals,Ffilter);
end

%% Add 29.975 Seconds to Start of Plant Measurements
padding_length = round(29.975/dt);
Plant = [zeros(padding_length,length(measurements));Plant];

%% Compute R as Covariance of Measurement Noise
R = cov(highFiltered);