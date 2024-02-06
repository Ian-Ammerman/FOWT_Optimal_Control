%% Simulate Non-Linear OpenFAST
clear all; close all; clc;

tests = {'FC4_Step_Wind'};

FASTdir = 'C:\\Umaine Google Sync\\Masters Working Folder\\FOCAL_C2';
model = 'FOCAL_C4';

for i = 1:length(tests)
    % Run OpenFAST
    runFAST(model,tests{i},FASTdir,'CheckSimFolder',false,'Version','v3_5_1');
end

%% Perform OpenFAST Linearization
clear all; close all; clc;
tests = {'000_Linearize'};
FASTdir = 'C:\\Umaine Google Sync\\Masters Working Folder\\FOCAL_C2';

% Linearization inputs
model = 'FOCAL_C4_Straight';
num_lin_times = 36; % Number of linearizations for rotor averaging
inflow_file_name = 'FOCAL_C4_InflowFile.dat';
inflow_line = 14; % line of inflow file for HWindSpeed
% wind_type_line = 5; % line of inflow file for WindType
wind_speed = 10; % constant wind speed for linearization
% wind_type = 0;

% Form inflow file directory
inflow_dir = sprintf('%s\\Models\\%s\\%s',FASTdir,model,inflow_file_name);

for i = 1:length(tests)
    setInflow(inflow_dir,inflow_line,wind_speed);
    runFAST(model,tests{i},FASTdir,'MoveFiles',false,'CheckSimFolder',false,'Version','v3_5_1');
end
%
cd(sprintf('%s\\Models\\%s\\Linear_Files',FASTdir,model));
processMBC3(num_lin_times,model)


% hydro_system = ReadFASTLinear('FOCAL_C4.1.HD.lin');
% 
% A = hydro_system.A;
% B = hydro_system.B;
% C = hydro_system.C;
% D = hydro_system.D;
% 
% Hydro_OP = hydro_system.x_op;
% 
% save('FOCAL_C4_HD_A.mat','A');
% save('FOCAL_C4_HD_B.mat','B');
% save('FOCAL_C4_HD_C.mat','C');
% save('FOCAL_C4_HD_D.mat','D');
% save('FOCAL_C4_Hydro_OP.mat','Hydro_OP');

%% Run State-Space Model
clear all; close all; clc;
% tests = {'FD_Surge','FD_Heave','FD_Pitch'}
tests = {'Test_04'};

% FD_IC = readmatrix('FD_IC.csv');
% FD_IC = FD_IC(:,2);

SLXdir = 'C:\Umaine Google Sync\Masters Working Folder\FOCAL_C2';
model = 'DT1_Locked';

for i = 1:length(tests)
    IC_x = zeros(20,1);
    % IC_x(i) = FD_IC(i);
    simout = runSLX(model,tests{i},SLXdir,'SeparateOutput',true,'InitialConditions',IC_x);
end

%% Run State-Space w/ Kalman Filter
clear all; close all; clc;

tests = {'Test_01'};
measurements = {'PtfmPitch','PtfmRoll','FAIRTEN1','FAIRTEN2','FAIRTEN3'};

SLXdir = 'C:\Umaine Google Sync\Masters Working Folder\FOCAL_C2';
model = 'DT1_Locked';

for i = 1:length(tests)
    IC_x = zeros(20,1);
    runSLX(model,tests{i},SLXdir,'Observer',true,'MeasurementFields',measurements,...
        'SeparateOutput',true,'InitialConditions',IC_x);
end