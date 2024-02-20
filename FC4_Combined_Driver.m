% FC4 - SS_Model_Driver
close all; clear all; clc
%% Load in Simulation or Test Data
load('Test_Results.mat','test_results');
load('OpenFAST_Results.mat','sim_results');
% test_results = sim_results;

%% Prepare Time Vector
% Extract time vector
test_time = test_results.Time;
sim_time = sim_results.Time;

% Smooth time vector
if exist("test_time","var")
    dt = mean(diff(test_time));
    test_time = linspace(min(test_time),max(test_time),length(test_time))';
end

% Prepare to shift wind input
if exist("test_time","var")
    dt = max(test_time)/length(test_time);
    causality_shift_index = floor(29.95/dt);
else
   dt = max(sim_time)/length(sim_time);
   causality_shift_index = floor(29.95/dt); 
end

%% Prepare Wave Input
% From Experiment
eta = test_results.Wave1Elev;
% eta = sim_results.Wave1Elev;

%% Prepare wind input
% Wind Case
wind_case = 3;

% Load in appropriate wind file
switch wind_case
    case 0 % No wind
        if exist("test_time","var")
            wind = zeros(size(test_time));
        else
            wind = zeros(size(sim_time));
        end
        rotor_lock = true;
    case 1 % W01 - Below Rated
        wind = readmatrix('C:\Umaine Google Sync\Masters Working Folder\FOCAL_C2\Models\FOCAL_C4\Wind\W01_fullScale_20230505.wnd');
        if exist("test_time","var")
            wind = pchip(wind(:,1),wind(:,2),test_time);
        else
            wind = pchip(wind(:,1),wind(:,2),sim_time);
        end
        
        rotor_lock = false;
    case 2 % W02 - Rated
        wind = readmatrix('C:\Umaine Google Sync\Masters Working Folder\FOCAL_C2\Models\FOCAL_C4\Wind\W02_fullScale_R02_20230606.wnd','FileType','text');
        if exist("test_time","var")
            wind = pchip(wind(:,1),wind(:,2),test_time);
        else
            wind = pchip(wind(:,1),wind(:,2),sim_time);
        end
        rotor_lock = false;
    case 3 % W03 - Above Rated
        wind = readmatrix('C:\Umaine Google Sync\Masters Working Folder\FOCAL_C2\Models\FOCAL_C4\Wind\W03_fullScale_R02_20230613.wnd','FileType','text');
        if exist("test_time","var")
            wind = pchip(wind(:,1),wind(:,2),test_time);
        else
            wind = pchip(wind(:,1),wind(:,2),sim_time);
        end
        rotor_lock = false;
    case 4 % Step Wind
        wind = readmatrix("C:\Umaine Google Sync\Masters Working Folder\FOCAL_C2\Models\FOCAL_Base\Wind\Step_Wind_N10_U1_T750.wnd",'FileType','text');
        if exist("test_time","var")
            wind = pchip(wind(:,1),wind(:,2),test_time);
        else
            wind = pchip(wind(:,1),wind(:,2),sim_time);
        end
        rotor_lock = false;
end

% Time-shift wind to account for hydro causalization time
% wind = [zeros(causality_shift_index,1);
%         wind(1:end-causality_shift_index)];

%% Prepare Control Input Values
% Blade pitch command (collective)
try
    c_pitch = test_results.pitch1Position*(pi/180);
catch
    c_pitch = sim_results.BldPitch1*(pi/180);
    disp('Using OpenFAST blade pitch values.');
end

% Generator torque command
try
    gen_torque = test_results.genTorqueSetpointActual;
catch
    gen_torque = sim_results.GenTq*10^3;
    disp('Using OpenFAST generator torque values.')
end

%% Form Input Vector
sim_inputs = [wind,gen_torque,c_pitch,eta];

%% Load in SS Model & Remove Azimuth State
% Define Path
platform_dir = 'C:\Umaine Google Sync\GitHub\FOWT_Optimal_Control\Models\FOCAL_C4\Linear_Files\3 - Combined Model';

% Load in raw files
load(sprintf('%s\\FOCAL_C4_A.mat',platform_dir),'A');
load(sprintf('%s\\FOCAL_C4_B.mat',platform_dir),'B');
load(sprintf('%s\\FOCAL_C4_C.mat',platform_dir),'C');
load(sprintf('%s\\FOCAL_C4_D.mat',platform_dir),'D');
load(sprintf('%s\\FOCAL_C4_y_OP.mat',platform_dir));

% Remove rotor azimuth state from state vector
A = A([1:8,10:end],[1:8,10:end]);
B = B([1:8,10:end],[1,8,9,94]);
C = C(:,[1:8,10:end]);
D = 0*D(:,[1,8,9,94]);

% Discretize Platform
platform_sys_c = ss(A,B,C,D);
platform_sys_d = c2d(platform_sys_c,dt,'zoh');
% [A_platform,B_platform,C_platform,D_platform] = ssdata(platform_sys_d);

% Reduce model size
R = reducespec(platform_sys_d,"modal");
% view(R)


rsys = getrom(R,Frequency=[0.05,1],Damping=[0,0.4]);

platform_sys_d = rsys;


% Clear out A,B,C,D matrices
clear A B C D platform_sys_c

%% Perform Simulation
if exist("test_time","var")
    Y = lsim(platform_sys_d,sim_inputs,test_time);
    ss_time = [test_time;0];
else
    Y = lsim(platform_sys_d,sim_inputs,sim_time);
    ss_time = sim_time;
end

Y = Y + y_OP';

%% Plot Results
% Plot parameters
tmax = 7500;

% Plot Platform Surge
figure
% subplot(4,1,2)
gca; hold on; box on;
title('Platform Surge')
xlim([0,tmax])
plot(ss_time(1:end-1)-29.95,Y(:,18),'DisplayName','State-Space')
plot(sim_time,sim_results.PtfmSurge,'DisplayName','OpenFAST')
try
    plot(test_time,test_results.PtfmSurge,'DisplayName','Experiment')
end
legend

% Plot Platform Heave
figure
% subplot(4,1,1)
gca; hold on; box on;
xlim([0,tmax])
title('Platform Heave [m]')
plot(ss_time(1:end-1)-29.95,Y(:,20),'DisplayName','State-Space');
plot(sim_time,sim_results.PtfmHeave,'DisplayName','OpenFAST')
try
    plot(test_time,test_results.PtfmHeave,'DisplayName','Experiment')
end
legend

% Plot Platform Pitch
figure
% subplot(4,1,1)
gca; hold on; box on;
xlim([0,tmax])
title('Platform Pitch [deg]')
plot(ss_time(1:end-1)-29.95,Y(:,22),'DisplayName','State-Space');
plot(sim_time,sim_results.PtfmPitch,'DisplayName','OpenFAST')
try
    plot(test_time,test_results.PtfmPitch,'DisplayName','Experiment')
end
legend

% Plot rotor speed
figure
% subplot(4,1,3)
gca; hold on; box on;
xlim([0,tmax])
title('Rotor Speed [RPM]')
% xlim([0 500])
plot(ss_time(1:end-1)-29.95,Y(:,9),'DisplayName','State-Space');
plot(sim_time,sim_results.RotSpeed,'DisplayName','OpenFAST')
try
    plot(test_time,(test_results.genSpeed*(30/pi)),'DisplayName','Experiment');
end
legend

% Plot tower bending
figure
gca; hold on; box on;
xlim([0,tmax]);
title('Tower Base Bending Moment')
plot(ss_time(1:end-1)-29.95,Y(:,13),'DisplayName','State-Space');
plot(test_time,test_results.towerBotMy*10^-3,'DisplayName','Experiment')
legend

% % Plot wave elevation
% figure
% % subplot(4,1,3)
% gca; hold on; box on;
% xlim([0,tmax])
% title('Wave Elevation [m]')
% plot(test_results.Time,test_results.Wave1Elev)
% 
% % Plot Wind Speed
% figure
% % subplot(4,1,4)
% gca; hold on; box on;
% plot(test_results.Time,test_results.WindCal,'DisplayName','Experiment')
% title('Wind Speed [m/s]')
% xlabel('Time [s]')
% legend
