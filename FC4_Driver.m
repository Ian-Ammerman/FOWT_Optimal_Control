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
    time = linspace(min(test_time),max(test_time),length(test_time))';
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
wind_case = 2;

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
wind = [zeros(causality_shift_index,1);
        wind(1:end-causality_shift_index)];

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

%% Load in Platform Model
% Define Path
platform_dir = 'C:\Umaine Google Sync\GitHub\FOWT_Optimal_Control\Models\FOCAL_C4\Linear_Files\8 - Platform (Reduced)';

% Load in raw files
load(sprintf('%s\\FOCAL_C4_A.mat',platform_dir),'A');
load(sprintf('%s\\FOCAL_C4_B.mat',platform_dir),'B');
load(sprintf('%s\\FOCAL_C4_C.mat',platform_dir),'C');
load(sprintf('%s\\FOCAL_C4_D.mat',platform_dir),'D');
% load(sprintf('%s\\FOCAL_C4_Output_OP.mat',platform_dir),'y_op');

% Remove rotor azimuth state from state vector
A = A([1:5,6:end],[1:5,6:end]);
B = B([1:5,6:end],[301,2107:2112,2195,2196,3943:3954]);
C = C(:,[1:5,6:end]);
D = 0*D(:,[301,2107:2112,2195,2196,3943:3954]);

% Discretize Platform
platform_sys_c = ss(A,B,C,D);
platform_sys_d = c2d(platform_sys_c,dt,'zoh');
[A_platform,B_platform,C_platform,D_platform] = ssdata(platform_sys_d);

% Clear out A,B,C,D matrices
clear A B C D platform_sys_d platform_sys_c

%% Load in Hydrodynamics Model (FC4)
% Define Path
hydro_dir = 'C:\Umaine Google Sync\Masters Working Folder\FOCAL_C2\Models\FOCAL_C4\Linear_Files\2 - Hydrodynamics';

% Load in raw files
load(sprintf('%s\\FOCAL_C4_HD_A.mat',hydro_dir),'A');
load(sprintf('%s\\FOCAL_C4_HD_B.mat',hydro_dir),'B');
load(sprintf('%s\\FOCAL_C4_HD_C.mat',hydro_dir),'C');
load(sprintf('%s\\FOCAL_C4_HD_D.mat',hydro_dir),'D');
load(sprintf('%s\\FOCAL_C4_Hydro_OP.mat',hydro_dir),'Hydro_OP');

% Convert Hydro_OP type
Hydro_OP = cell2mat(Hydro_OP);

% Trim inputs
B = B(:,[37,7,8,9,10,11,12]);
D = 0*D(:,[37,7,8,9,10,11,12]);

% Discretize hydrodynamics model
hydro_sys_c = ss(A,B,C,D);
hydro_sys_d = c2d(hydro_sys_c,dt,'zoh');
[A_hydro,B_hydro,C_hydro,D_hydro] = ssdata(hydro_sys_d);

% Clear extra variables
clear A B C D hydro_sys_c hydro_sys_d

%% Simulate System
% Initialization (zero IC)
if exist('test_time','var')
    ss_time = test_time;
else
    ss_time = sim_time;
end
x_HD = zeros(size(A_hydro,1),1);
x = zeros(size(A_platform,1),1);
Y = zeros(size(C_platform,1),length(ss_time)-1);
platform_positions = zeros(6,1);
platform_velocities = zeros(6,1);
X_log = zeros(size(x,1),length(ss_time)-1);


% Loop over simulation time
for i = 1:length(ss_time)-1

    % Separate platform position/velocity
    platform_positions([1,3,5]) = x([1,2,3]);
    platform_velocities([1,3,5]) = x([6,7,8]);

    % Define HydroDyn Input
    u_hydro = [eta(i);
               platform_velocities];

    % Update HydroDyn States
    x_HD = A_hydro*x_HD + B_hydro*u_hydro;

    % Extract resultant forces for platform input
    hydro_out = C_hydro*x_HD;
    platform_forces = hydro_out(2:end);
    % platform_forces = hydro_out;

    % Form platform input vector
    u_platform = [wind(i);
                  platform_forces;
                  gen_torque(i);
                  c_pitch(i);
                  platform_positions;
                  platform_velocities;];

    % Update platform states
    x = A_platform*x + B_platform*u_platform;

    % Lock rotor for wave-only case
    if rotor_lock == true
        x(end) = 0;
    end

    % Store platform outputs
    Y(:,i) = C_platform*x;
    X_log(:,i) = x;
end

Y = Y';

%% Plot Results
close all;
% Plot parameters
tmax = 7500;

% Plot Platform Surge
figure
% subplot(4,1,2)
gca; hold on; box on;
title('Platform Surge')
xlim([0,tmax])
plot(ss_time(1:end-1)-29.95,Y(:,14),'DisplayName','State-Space')
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
plot(ss_time(1:end-1)-29.95,Y(:,16),'DisplayName','State-Space');
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
plot(ss_time(1:end-1)-29.95,Y(:,18),'DisplayName','State-Space');
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
plot(ss_time(1:end-1)-29.95,Y(:,5),'DisplayName','State-Space');
plot(sim_time,sim_results.RotSpeed,'DisplayName','OpenFAST')
try
    plot(test_time,(test_results.genSpeed*(30/pi)),'DisplayName','Experiment');
end
legend

% Plot wave elevation
figure
% subplot(4,1,3)
gca; hold on; box on;
xlim([0,tmax])
title('Wave Elevation [m]')
plot(ss_time,eta)

% Plot Wind Speed
figure
% subplot(4,1,4)
gca; hold on; box on;
plot(ss_time,wind,'DisplayName','Experiment')
title('Wind Speed [m/s]')
xlabel('Time [s]')
legend

figure
gca; hold on;
plot(ss_time(1:end-1)-29.95,Y(:,9),'DisplayName','State-Space');
plot(test_results.Time,test_results.towerBotMy*10^-3,'DisplayName','Experiment');
title('Tower Base Bending Moment')
legend

%% Prepare Process Covariance (Q) Matrix
% Transpose states
x_process = X_log';
x_process(:,4:6) = x_process(:,4:6)*57.3;
x_process(:,10:12) = x_process(:,10:12)*57.3;
x_process(:,13) = x_process(:,13)*9.549296;

x_process = x_process(50000:end,:);

% State-space standard deviations
ss_std = std(x_process);
ss_mean = mean(x_process);

% Corresponding test data
x_test(:,1) = test_results.PtfmSurge;
x_test(:,2) = test_results.PtfmSway;
x_test(:,3) = rMean(test_results.PtfmHeave);
% x_test(:,3) = test_results.PtfmHeave;
x_test(:,4) = test_results.PtfmRoll;
x_test(:,5) = test_results.PtfmPitch;
x_test(:,6) = test_results.PtfmYaw;

x_test(2:end,7:12) = (x_test(2:end,1:6) - x_test(1:end-1,1:6))./0.0416;
x_test(1,7:12) = x_test(2,7:12);

x_test(:,13) = test_results.genSpeed*(30/pi);

x_test = x_test(50000:end,:);

%% Form P matrix
x_test_filtered = highpass(x_test,1,24);
P = diag(var(x_test_filtered));

figure
plot(test_results.Time(50000:end),x_test(:,3))
hold on
plot(test_results.Time(50000:end),x_test_filtered(:,3))
plot(test_results.Time(50000:end),x_test(:,3)-x_test_filtered(:,3))

%% Form Q Matrix

%%% Compute uncertainty of all states directly
% Experimental standard deviations
exp_std = std(x_test);
exp_mean = mean(x_test);

mean_diff = abs(exp_mean - ss_mean);

% Compute difference between STDs
diff_std = abs(ss_std - exp_std);

% Form Q matrix
Q_diag = diag(2*diff_std);
Q = Q_diag;

%%% Compute uncertainty of measurements & project onto states
% % Measurements from linear system
% y_measurements = Y(:,[29,31,33,32,11,56,57,58,19]);
% y_measurements = y_measurements(50000:end,:);
% 
% % Measurements from experiment
% exp_measurements = [test_results.PtfmSurge,...
%                     test_results.PtfmHeave,...
%                     test_results.PtfmPitch,...
%                     test_results.PtfmRoll,...
%                     test_results.genSpeed*(30/pi),...
%                     test_results.leg1MooringForce,...
%                     test_results.leg2MooringForce,...
%                     test_results.leg3MooringForce,...
%                     test_results.towerBotMy*10^-3];
% 
% exp_measurements = exp_measurements(50000:end,:);
% 
% % Compute standard deviations
% ss_std = std(y_measurements);
% exp_std = std(exp_measurements);
% 
% ss_mean = mean(y_measurements);
% exp_mean = mean(exp_measurements);
% 
% diff_std = abs(ss_std - exp_std);
% 
% Qc = diag(diff_std);
% 
% H = C_platform([29,31,33,32,11,56,57,58,19],:);
% 
% Q = H'*Qc*H;

