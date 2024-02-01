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
platform_dir = 'C:\Umaine Google Sync\Masters Working Folder\FOCAL_C2\Models\FOCAL_C4\Linear_Files\7 - Platform (Rigid)';

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

% Scale outputs
C(34:end,:) = C(34:end,:)*10^-3;
C(8:10,:) = C(8:10,:) * 10^-3;

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
D = D(:,[37,7,8,9,10,11,12]);

% Discretize hydrodynamics model
hydro_sys_c = ss(A,B,C,D);
hydro_sys_d = c2d(hydro_sys_c,dt,'zoh');
[A_hydro,B_hydro,C_hydro,D_hydro] = ssdata(hydro_sys_d);

% Clear extra variables
clear A B C D hydro_sys_c hydro_sys_d

%% Define System Measurements for Correction
% Measurements:
% --------------
% 1) Tower FA Bending Moment (Strain Gauges)
% 2) Tower SS Bending Moment (Strain Gauges)
% 3) Rotor Speed (SCADA)
% 4) Mooring Tensions (x3) (LCs or strain gauge array?)

% Combine measurements to single matrix
% FA_bending = test_results.towerBotMy*10^-6;
% SS_bending = test_results.towerBotMx*10^-6;
pitch = test_results.PtfmPitch;
roll = test_results.PtfmRoll;
rotor_speed = test_results.genSpeed*(30/pi);
mooring_tension_1 = test_results.leg1MooringForce*10^-3;
mooring_tension_2 = test_results.leg2MooringForce*10^-3;
mooring_tension_3 = test_results.leg3MooringForce*10^-3;

% system_measurements = [FA_bending,SS_bending,rotor_speed,mooring_tension_1,mooring_tension_2,mooring_tension_3];
system_measurements = [pitch,roll,rotor_speed,mooring_tension_1,mooring_tension_2,mooring_tension_3];

% Form measurement function (H) from SS output
% H = C_platform([9,8,4,34,35,36],:); % tower bending
H = C_platform([18,17,4,34,35,36],:); % angular displacement

%% Compute Measurement Covariance Matrix
% % Remove mean from measurements
% for i = [1,2,4,5,6]
%     system_measurements(:,i) = system_measurements(:,i) - mean(system_measurements(75000:end,i));
% end

% Low-pass filter @ 6Hz
f_sample = length(test_results.Time)/max(test_results.Time);
filtered_measurements = lowpass(system_measurements,1,f_sample);
% filtered_measurements = system_measurements;
filtered_measurements = [getRamp(zeros(1,size(system_measurements,2)),filtered_measurements(1,:),714);filtered_measurements];

measurement_noise = highpass(system_measurements,1,f_sample);

% Covariance of measurements
measurement_covariance = cov(measurement_noise(23906:end-1000,:));
R = measurement_covariance;

R(1,1) = 0.0031;
R(2,2) = 0.0031;

R(4,4) = 2.6*10^6;
R(5,5) = 2.6*10^6;
R(6,6) = 2.6*10^6;

%% Load in P & Q Matrices
kalman_dir = 'C:\Umaine Google Sync\GitHub\FOWT_Optimal_Control\Models\FOCAL_C4\Linear_Files\5 - Kalman Files';

load(sprintf('%s\\Q_stiff.mat',kalman_dir));
load(sprintf('%s\\P_stiff.mat',kalman_dir));

% Adjust Q values
q_adj_gain = 500;
q_adj_points = [2,4,6,8,10,12];

for i = q_adj_points
    Q(i,i) = q_adj_gain*Q(i,i);
end

%% Simulate System (Kalman Filter)
% Initialization (zero IC)
if exist('test_time','var')
    ss_time = test_time;
else
    ss_time = sim_time;
end
x_HD = zeros(size(A_hydro,1),1);
x = zeros(size(A_platform,1),1);
Y = zeros(size(C_platform,1),length(ss_time)-1);

% Loop over simulation time
for i = 1:length(ss_time)-1

    % Separate platform position/velocity
    platform_positions = x(1:6);
    platform_velocities = x(7:12);

    % Define HydroDyn Input
    u_hydro = [eta(i);
               platform_velocities];

    % Update HydroDyn States
    x_HD = A_hydro*x_HD + B_hydro*u_hydro;

    % Extract resultant forces for platform input
    hydro_out = C_hydro*x_HD + D_hydro*u_hydro;
    platform_forces = hydro_out(2:end);
    % platform_forces = hydro_out;

    % Form platform input vector
    u_platform = [wind(i);
                  platform_forces;
                  gen_torque(i);
                  c_pitch(i);
                  platform_positions;
                  platform_velocities;];

    % Do prediction step
    [x,P] = predict(x,P,A_platform,dGain(1,Q),B_platform,u_platform);

    % Get "measurements"
    z = filtered_measurements(i,:);

    % Do update step
    [x,P,K] = update(H,P,R,z',x);

    % Lock rotor for wave-only case
    if rotor_lock == true
        x(end) = 0;
    end

    % Store platform outputs
    Y(:,i) = C_platform*x;
end

Y = Y';

% Save P for inspection
Pk = P;

%% Simulate System (State-Space)
% Initialization (zero IC)
if exist('test_time','var')
    ss_time = test_time;
else
    ss_time = sim_time;
end
x_HD = zeros(size(A_hydro,1),1);
x = zeros(size(A_platform,1),1);
Y_raw = zeros(size(C_platform,1),length(ss_time)-1);
platform_positions = zeros(6,1);
platform_velocities = zeros(6,1);


% Loop over simulation time
for i = 1:length(ss_time)-1

    % Separate platform position/velocity
    platform_positions = x(1:6);
    platform_velocities = x(7:12);

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

    % Do prediction step
    [x,P] = predict(x,P,A_platform,dGain(1,Q),B_platform,u_platform);

    % Lock rotor for wave-only case
    if rotor_lock == true
        x(end) = 0;
    end

    % Store platform outputs
    Y_raw(:,i) = C_platform*x;
end

Y_raw = Y_raw';

%% Plot Results
% Plot parameters
tmax = 7500;

% Plot Platform Surge
figure
% subplot(4,1,2)
gca; hold on; box on;
title('Platform Surge')
xlim([0,tmax])
plot(ss_time(1:end-1)-29.95,rMean(Y_raw(:,14)),'DisplayName','State-Space')
plot(ss_time(1:end-1)-29.95,rMean(Y(:,14)),'DisplayName','Kalman')
% plot(sim_time,sim_results.PtfmSurge,'DisplayName','OpenFAST')
try
    plot(test_time,rMean(test_results.PtfmSurge),'DisplayName','Experiment')
end
legend

% Plot Platform Heave
figure
% subplot(4,1,1)
gca; hold on; box on;
xlim([0,tmax])
title('Platform Heave [m]')
plot(ss_time(1:end-1)-29.95,rMean(Y_raw(:,16)),'DisplayName','State-Space');
plot(ss_time(1:end-1)-29.95,rMean(Y(:,16)),'DisplayName','Kalman');
% plot(sim_time,sim_results.PtfmHeave,'DisplayName','OpenFAST')
try
    plot(test_time,rMean(test_results.PtfmHeave),'DisplayName','Experiment')
end
legend

% Plot Platform Pitch
figure
% subplot(4,1,1)
gca; hold on; box on;
xlim([0,tmax])
title('Platform Pitch [deg]')
plot(ss_time(1:end-1)-29.95,rMean(Y_raw(:,18)),'DisplayName','State-Space');
plot(ss_time(1:end-1)-29.95,rMean(Y(:,18)),'DisplayName','Kalman');
% plot(sim_time,sim_results.PtfmPitch,'DisplayName','OpenFAST')
try
    plot(test_time,rMean(test_results.PtfmPitch),'DisplayName','Experiment')
end
legend

% Plot tower fore-aft bending moment
figure
gca; hold on; box on;
xlim([0,tmax])
title('Tower FA Bending Moment [kN-m]')
plot(ss_time(1:end-1)-29.95,rMean(Y_raw(:,9)),'DisplayName','State-Space');
plot(ss_time(1:end-1)-29.95,rMean(Y(:,9)),'DisplayName','Kalman');
% plot(sim_time,sim_results.TwrBsMyt,'DisplayName','OpenFAST');
try
    plot(test_time,rMean(test_results.towerBotMy*10^-6),'DisplayName','Experiment')
end
legend

% Plot rotor speed
figure
% subplot(4,1,3)
gca; hold on; box on;
xlim([0,tmax])
title('Rotor Speed [RPM]')
% xlim([0 500])
plot(ss_time(1:end-1)-29.95,Y_raw(:,5),'DisplayName','State-Space');
plot(ss_time(1:end-1)-29.95,Y(:,5),'DisplayName','Kalman');
% plot(sim_time,sim_results.RotSpeed,'DisplayName','OpenFAST')
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


%% Functions --------------------------------------------------------- %%
% Prediction (Labbe, 2020, pg 212)
function [x,P] = predict(x,P,F,Q,B,u)
    x = F*x + B*u; %predict states
    P = F*P*F' + Q; %predict process covariance
end

% Update
function [x,P,K] = update(H,P,R,z,x)
    S = H*P*H' + R; % Project system uncertainty into measurement space & add measurement uncertainty
    K = P*H'*inv(S);
    y = z-H*x; % Error term
    x = x+K*y;
    KH = K*H;
    P = (eye(size(KH))-KH)*P;
end

% Mean relative error
function e = MRE(model,experiment)
    
    % Remove mean from each
    model = rMean(model);
    experiment = rMean(experiment);

    % Compute mean relative error
    e = mean(abs(model-experiment)/(max(experiment)-min(experiment)));
end