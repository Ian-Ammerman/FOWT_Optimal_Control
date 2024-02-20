% FC4 - SS_Model_Driver
close all; clear all; clc

%% Set Top-Level Linear directory
linear_dir = 'C:\Umaine Google Sync\GitHub\FOWT_Optimal_Control\Models\FOCAL_C4\Linear_Files';

%% Load in Simulation or Test Data
load('Test_Results.mat','test_results');
load('OpenFAST_Results.mat','sim_results');

%% Prepare Time Vector
% Form vector with uniform spacing
dt = 0.025;
f_sample = 1/dt; % useful later
num_steps = floor(max(test_results.Time)/dt); % ensure evenly divisible by new_dt
new_time = transpose(linspace(0,max(test_results.Time),num_steps));
test_time = new_time;

% Replace test results time vector with new
old_time = test_results.Time;
test_results.Time = new_time;

%% Adjust Test Results to Match New Time Vector
fields = fieldnames(test_results);
for i = 2:length(fields)
    test_results.(fields{i}) = pchip(old_time,test_results.(fields{i}),new_time);
end

%% Prepare Wave Input
eta = test_results.Wave1Elev;

%% Prepare wind input
% Wind Case
wind_case = 3;

% Load wind file
wind_file_path = 'C:\Umaine Google Sync\GitHub\FOWT_Optimal_Control\Wind_Files';
wind = getFOCALWindVector(wind_file_path,wind_case,test_time);

%% Prepare Control Input Values
% Blade pitch command (collective)
c_pitch = (pi/180)*test_results.pitch1Position;

% Ind. Pitch
idv_pitch = zeros(size(c_pitch,1),3);

% Generator torque command
gen_torque = test_results.genTorqueSetpointActual;

%% Load in Platform Model
% Define Path
platform_folder = '3 - Combined Model';
platform_dir = sprintf('%s\\%s',linear_dir,platform_folder);

% Define state & input ranges
state_range = [1:8,10:368];
control_range = [1,4,5,6,8,9,94];

% Load in model matrices
[A_platform,B_platform,C_platform,D_platform,x_OP,y_OP] = loadPlatformModel(platform_dir,state_range,control_range,dt);

%% Define System Measurements for Correction
% Combine measurements to single matrix
pitch = test_results.PtfmPitch;
roll = test_results.PtfmRoll;
rotor_speed = test_results.genSpeed*(30/pi);
FA_nacelle_acceleration = test_results.accelNacelleAx;

system_measurements = [pitch,roll,rotor_speed,FA_nacelle_acceleration];

% Form measurement function (H) from SS output
H = C_platform([22,21,8,32],:); % angular displacement
h_OP = y_OP([22,21,8,32]);

clear pitch roll rotor_speed FA_nacelle_acceleration

%% Compute Measurement Covariance Matrix
% Low-pass filter measurements to improve KF performance
filtered_measurements = lowpass(system_measurements,1,f_sample);

% Extract measurement noise
measurement_noise = highpass(system_measurements,1,f_sample);

% Covariance of measurements
measurement_covariance = cov(measurement_noise(23906:end-1000,:));
R = measurement_covariance;
% R = 0.001*eye(size(measurement_covariance));

% R(1,1) = 0.0031;
% R(2,2) = 0.0031;
% R(4,4) = 0.4;
% R(5,5) = 0.4;
% R(6,6) = 0.65;
% R(7,7) = 0.04;

%% Load in P & Q Matrices
% File location
kalman_dir = 'C:\Umaine Google Sync\GitHub\FOWT_Optimal_Control\Models\FOCAL_C4\Linear_Files\5 - Kalman Files';

% Load in values
% load(sprintf('%s\\FC4_Q.mat',kalman_dir));
% load(sprintf('%s\\FC4_P.mat',kalman_dir));

% Set Q as identity
Q = eye(size(A_platform));

% Set P as zero matrix
P = zeros(size(A_platform));

%% Simulate System (Kalman Filter)
disp('Beginning Kalman filter simulation...')

% Initialization (zero IC)
ss_time = test_time;
x = zeros(size(A_platform,1),1);
Y = zeros(size(C_platform,1),length(ss_time));
etime = zeros(size(Y,2),1);

% Loop over simulation time
for i = 1:length(ss_time)
    tic
    % Form platform input vector
    u_platform = [wind(i);
                  idv_pitch(i,:)';
                  gen_torque(i);
                  c_pitch(i);
                  eta(i)];

    % Do prediction step
    [x,P] = predict(x,P,A_platform,Q,B_platform,u_platform);

    % Get "measurements"
    z = filtered_measurements(i,:);

    % Do update step
    [x,P,K] = update(H,P,R,z',x,h_OP);

    % Store platform outputs
    Y(:,i) = C_platform*x + y_OP;
    % etime(i) = toc;
end

Y = Y';

% Save P for inspection
Pk = P;

%% Simulate System (State-Space)
disp('Beginning state-space simulation...')

% Initialization (zero IC)
x = zeros(size(A_platform,1),1);
Y_raw = zeros(size(C_platform,1),length(ss_time));

% Loop over simulation time
for i = 1:length(ss_time)-1

    % Form platform input vector
    u_platform = [wind(i);
                  idv_pitch(i,:)';
                  gen_torque(i);
                  c_pitch(i),
                  eta(i)];

    % Do prediction step
    [x,P] = predict(x,P,A_platform,dGain(1,Q),B_platform,u_platform);

    % Store platform outputs
    Y_raw(:,i) = C_platform*x + y_OP;
end

Y_raw = Y_raw';

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
% plot(ss_time(1:end-1)-29.95,Y_raw(:,18),'DisplayName','State-Space')
plot(ss_time(1:end),Y(:,18),'DisplayName','Kalman')
% plot(sim_time,sim_results.PtfmSurge,'DisplayName','OpenFAST')
try
    plot(test_time,test_results.PtfmSurge,'DisplayName','Experiment')
end
legend

% Plot Platform Sway
figure
% subplot(4,1,2)
gca; hold on; box on;
title('Platform Sway')
xlim([0,tmax])
% plot(ss_time(1:end-1)-29.95,Y_raw(:,19),'DisplayName','State-Space')
plot(ss_time(1:end),Y(:,19),'DisplayName','Kalman')
% plot(sim_time,sim_results.PtfmSurge,'DisplayName','OpenFAST')
try
    plot(test_time,test_results.PtfmSway,'DisplayName','Experiment')
end
legend

% Plot Platform Heave
figure
% subplot(4,1,1)
gca; hold on; box on;
xlim([0,tmax])
title('Platform Heave [m]')
% plot(ss_time(1:end-1)-29.95,Y_raw(:,20),'DisplayName','State-Space');
plot(ss_time(1:end),Y(:,20),'DisplayName','Kalman');
% plot(sim_time,sim_results.PtfmHeave,'DisplayName','OpenFAST')
try
    plot(test_time,test_results.PtfmHeave,'DisplayName','Experiment')
end
legend

% Plot Platform Roll
figure
% subplot(4,1,2)
gca; hold on; box on;
title('Platform Roll')
xlim([0,tmax])
% plot(ss_time(1:end-1)-29.95,Y_raw(:,21),'DisplayName','State-Space')
plot(ss_time(1:end),Y(:,21),'DisplayName','Kalman')
% plot(sim_time,sim_results.PtfmSurge,'DisplayName','OpenFAST')
try
    plot(test_time,test_results.PtfmRoll,'DisplayName','Experiment')
end
legend

% Plot Platform Pitch
figure
% subplot(4,1,1)
gca; hold on; box on;
xlim([0,tmax])
title('Platform Pitch [deg]')
% plot(ss_time(1:end-1)-29.95,Y_raw(:,22),'DisplayName','State-Space');
plot(ss_time(1:end),Y(:,22),'DisplayName','Kalman');
% plot(sim_time,sim_results.PtfmPitch,'DisplayName','OpenFAST')
try
    plot(test_time,test_results.PtfmPitch,'DisplayName','Experiment')
end
legend

% Plot Platform Yaw
figure
% subplot(4,1,2)
gca; hold on; box on;
title('Platform Yaw')
xlim([0,tmax])
% plot(ss_time(1:end-1)-29.95,Y_raw(:,23),'DisplayName','State-Space')
plot(ss_time(1:end),Y(:,23),'DisplayName','Kalman')
% plot(sim_time,sim_results.PtfmSurge,'DisplayName','OpenFAST')
try
    plot(test_time,test_results.PtfmYaw,'DisplayName','Experiment')
end
legend

% Plot tower fore-aft bending moment
figure
gca; hold on; box on;
xlim([0,tmax])
title('Tower FA Bending Moment [kN-m]')
% plot(ss_time(1:end-1)-29.95,Y_raw(:,13),'DisplayName','State-Space');
plot(ss_time(1:end),Y(:,13),'DisplayName','Kalman');
% plot(sim_time,sim_results.TwrBsMyt,'DisplayName','OpenFAST');
try
    plot(test_time,test_results.towerBotMy*10^-3,'DisplayName','Experiment')
end
legend

% % Plot tower bending spectrum
% tpsd = myPSD(Y(:,13),f_sample,25);
% epsd = myPSD(test_results.towerBotMy*10^-6,f_sample,25);
% spsd = myPSD(Y_raw(:,13),f_sample,25);
% rat = tpsd(:,2)./epsd(1:end-1,2);
% diffpsd = tpsd(:,2)/mean(rat(1405:2071));
% figure; gca; hold on;
% title('Tower Bending PSD')
% plot(spsd(:,1),spsd(:,2),'DisplayName','State-Space');
% plot(tpsd(:,1),tpsd(:,2),'DisplayName','Kalman Filter'); 
% plot(epsd(:,1),epsd(:,2),'DisplayName','Experiment');
% plot(tpsd(:,1),diffpsd,'DisplayName','Kalman Scaled');
% xlim([0,0.2]); 
% ylim([0,6.6*10^5]);
% legend

% Plot rotor speed
figure
% subplot(4,1,3)
gca; hold on; box on;
xlim([0,tmax])
title('Rotor Speed [RPM]')
% xlim([0 500])
% plot(ss_time(1:end-1)-29.95,Y_raw(:,8),'DisplayName','State-Space');
plot(ss_time(1:end),Y(:,8),'DisplayName','Kalman');
% plot(sim_time,sim_results.RotSpeed,'DisplayName','OpenFAST')
try
    plot(test_time,(test_results.genSpeed*(30/pi)),'DisplayName','Experiment');
end
legend

% Plot lead mooring tension
figure
gca; hold on; box on;
title('Mooring Tension (1)')
% plot(ss_time(1:end-1)-29.95,Y_raw(:,44),'DisplayName','State-Space');
plot(ss_time(1:end),Y(:,44),'DisplayName','Kalman Filter');
plot(test_time,test_results.leg1MooringForce-3.28*10^5,'DisplayName','Experiment');
legend

% Plot lead mooring tension
figure
gca; hold on; box on;
title('Mooring Tension (2)')
% plot(ss_time(1:end-1)-29.95,Y_raw(:,45),'DisplayName','State-Space');
plot(ss_time(1:end)-29.95,Y(:,45),'DisplayName','Kalman Filter');
plot(test_time,test_results.leg2MooringForce-3.17*10^5,'DisplayName','Experiment');
legend

% Plot lead mooring tension
figure
gca; hold on; box on;
title('Mooring Tension (3)')
% plot(ss_time(1:end-1)-29.95,Y_raw(:,46),'DisplayName','State-Space');
plot(ss_time(1:end),Y(:,46),'DisplayName','Kalman Filter');
plot(test_time,test_results.leg3MooringForce-3.4*10^5,'DisplayName','Experiment');
legend

% Plot nacelle acceleration
figure
gca; hold on; box on;
title('Nacelle FA Acceleration')
% plot(ss_time(1:end-1)-29.95,Y_raw(:,32),'DisplayName','State-Space');
plot(ss_time(1:end),Y(:,32),'DisplayName','Kalman Filter');
plot(test_time,test_results.accelNacelleAx,'DisplayName','Experiment');
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

%% Clear window
clc

%% Functions --------------------------------------------------------- %%
% Prediction (Labbe, 2020, pg 212)
function [x,P] = predict(x,P,F,Q,B,u)
    x = F*x + B*u; %predict states
    P = F*P*F' + Q; %predict process covariance
end

% Update
function [x,P,K] = update(H,P,R,z,x,OP)
    S = H*P*H' + R; % Project system uncertainty into measurement space & add measurement uncertainty
    K = P*H'*inv(S);
    y = z-(H*x + OP); % Error term
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