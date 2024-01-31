%% Trim FC4 Linear Model
% Inputs to Keep:
% IFW 1:3
% ED Blade Pitch Commands (ind. & collective)
% 

%% Load in B Matrix
load('FOCAL_C4_B.mat','B');

%% Define Columns to Keep
keeps = [301:303,2107:2112,2191:2193];