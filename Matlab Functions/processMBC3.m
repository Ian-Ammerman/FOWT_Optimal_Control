function processMBC3(linTimes,model)

% Script to generate A,B,C,D matrix .mat files from .lin files via MBC3
% clear all; close all; clc;

%% Compile filenames into cell array
% linTimes = 36;
for i = 1:linTimes
    filenames{i} = sprintf('%s.%i.lin',model,i);
end

%% Perform coordinate transform
[mbc_data, matData, FAST_linData] = fx_mbc3(filenames);

%% Extract matrices from mbc_data
A = mbc_data.AvgA;
B = mbc_data.AvgB;
C = mbc_data.AvgC;
D = mbc_data.AvgD;

%% Extract Operation Point
x_OP = matData.Avgxop;

output_op = FAST_linData.y_op;
y_OP = cell2mat(output_op);

%% Save in .mat files
save('FOCAL_C4_A.mat','A');
save('FOCAL_C4_B.mat','B');
save('FOCAL_C4_C.mat','C');
save('FOCAL_C4_D.mat','D');
save('FOCAL_C4_X_OP.mat','x_OP')
save('FOCAL_C4_Y_OP.mat','y_OP')
