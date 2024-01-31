function SimulationOutput = runSLX(root,test,SLXdir,varargin)

%
% Written By: Ian Ammerman
% Written: 9/7/23
%
% runSLX runs a simulink simulation. Note that all directories are
% relative to the SLXdir. The simulation time is dependant on the time
% available in the wave elevation input file.
%
% Inputs:
% ----------
% model - name of model to test
% test - name of test folder for saving
% SLXdir - home directory (absolute path) of the driver
% buffer - settling time to add to start of simulation
% TMax - max time to process from wave/wind files (to be deprecated)

% -------------------------------------------------------------------- %

p = inputParser;
addRequired(p,'root',@ischar);
addRequired(p,'test',@ischar);
addRequired(p,'SLXdir');
addParameter(p,'Observer',false,@islogical);
addParameter(p,'MeasurementFields',{},@iscellstr);
addParameter(p,'Buffer',0,@isPositiveIntegerValuedNumeric);
addParameter(p,'SeparateOutput',false,@islogical);
addParameter(p,'PlatformInputs',[37:42],@ismatrix);
addParameter(p,'InitialConditions',zeros(20,1),@ismatrix);

parse(p,root,test,SLXdir,varargin{:});

root = p.Results.root;
test = p.Results.test;
SLXdir = p.Results.SLXdir;
Observer = p.Results.Observer;
MeasurementFields = p.Results.MeasurementFields;
Buffer = p.Results.Buffer;
SeparateOutput = p.Results.SeparateOutput;
PlatformInputs = p.Results.PlatformInputs;
IC_x = p.Results.InitialConditions;

simulink_model = 'Linear_wObserver';
cd(SLXdir)

%% Define & Load In Models
HD_path = sprintf('Models\\%s_HD',root);
Platform_path = sprintf('Models\\%s_Platform',root);
if SeparateOutput == true
    Output_Path = sprintf('Models\\%s_Platform_out',root);
end

home = pwd;

% ---------- HydroDyn ---------- %
cd(HD_path)

load(sprintf('%s_HD_A',root));
load(sprintf('%s_HD_B',root));
load(sprintf('%s_HD_C',root));
load(sprintf('%s_HD_D',root));
load(sprintf('%s_HD_ss_data.mat',root),'SS_data');

A_HD = A; B_HD = B(:,[37]); 
C_HD = C; D_HD = D(:,[37]);
% C_feedback = [zeros(6,10),eye(6),zeros(6,4)];
z7 = zeros(1,7);
mixmat = [1,zeros(1,6);
          z7;
          0, 1, zeros(1,5);
          z7;
          0, 0, 1, zeros(1,4);
          z7]
C_feedback = [zeros(6,7),mixmat]

cd(SLXdir)
% ---------- Platform ---------- %
cd(Platform_path)

load(sprintf('%s_Platform_A',root));
load(sprintf('%s_Platform_B',root));
load(sprintf('%s_Platform_C',root));
load(sprintf('%s_Platform_D',root));
load(sprintf('%s_Platform_ss_data.mat',root),'SS_data');

B = B(:,[37:42]);
y_op = SS_data.y_op

% Define Model
A_platform = A; B_platform = B; C_platform = C; D_platform = zeros(height(C_platform),width(B_platform));

cd(SLXdir);
% ---------- Output Model ---------- %
if SeparateOutput == true
    cd(Output_Path)

    load(sprintf('%s_Platform_out_C',root));
    load(sprintf('%s_Platform_out_ss_data.mat',root),'SS_data');

    C_out = C;
    % C_out = eye(20);
    %     C_out(5,5) = 57.3;
    %     C_out(4,4) = 57.3;
    %     C_out(6,6) = 57.3;
    %     C_out(14,14) = 57.3;
    %     C_out(15,15) = 57.3;
    %     C_out(16,16) = 57.3;
    cd(SLXdir)
else
    C_out = C_platform;
    % C_out = eye(20);
end

%% Define Output Names
y_desc = SS_data.y_desc;

string_new = cell(height(y_desc),3);
for i = 1:height(y_desc)
    string = y_desc{i};
    string_new = strsplit(string);

    state_name = string_new(2);
    state_name = strrep(state_name,',',''); % remove commas from names
    state_name = strrep(state_name,'[','');
    state_name = strrep(state_name,']','');
    out_state_names(1,i) = state_name;
end
outputNames = horzcat(out_state_names);

% outputNames = {'PtfmSurge','PtfmSway','PtfmHeave','PtfmRoll','PtfmPitch','PtfmYaw','TwrMode1FA','TwrMode1SS','TwrMode2FA','TwrMode2SS',...
%                'SurgeVelocity','SwayVelocity','HeaveVelocity','RollVelocity','PitchVelocity','YawVelocity','TwrMode1FAVel','TwrMode1SSVel','TwrMode2FAVel',...
%                'TwrMode2SSVel'};

%% Define Simulation Time &  HydroDyn Inputs
cd(sprintf('Simulations\\%s',test));

% Load in test results
try
    load('Test_Results.mat','test_results')
catch
    error('Test results not found in simulation directory :((');
end

% HydroDyn model inputs
try
    tstop = dsearchn(test_results.Time,25);
    eta = lowpass(test_results.Wave1Elev,1,24);
    eta = eta - mean(eta(1:tstop));
    time = test_results.Time;
    HD_input = [time,eta];
catch
    time = test_results.Time;
    eta = zeros(size(time));
    HD_input = [time,eta];
    % error(['Time and/or wave elevation not found in test results. Ensure structure field names are...' ...
        % 'formatted as ''Time'' and ''Wave1Elev'' accordingly and retry.']);
end

% Initial Conditions
% IC_x = [0,3.1787,-1.9769,0.0086,-0.4672,-0.2294,0.0204,zeros(1,14)];

%% Load in Plant 'Measurements'
if Observer == false
    Plant = zeros(1,height(C_platform)+1);
else
    Plant = zeros(length(time),length(MeasurementFields)+1);
    Plant(:,1) = time+29.975;
    Fs = length(time)/max(time)
    Ffilter = 2;
    FHfilter = 0.04;

    for i = 1:length(MeasurementFields)
        try
            if strcmp(MeasurementFields{i},'TwrBsMyt')
                vals = test_results.(MeasurementFields{i})*10^-3;
            else
                vals = test_results.(MeasurementFields{i});
            end
            baseline = vals-y_op{i};

        catch
            error(sprintf('Could not load %s from test results.',MeasurementFields{i}));
        end

        lowFiltered = lowpass(baseline,Ffilter,Fs);

        Plant(:,i+1) = lowFiltered;

        filtered(:,i) = lowFiltered;
        noise(:,i) = baseline-lowFiltered;
    end
end
cd(home)

%% Prepare Observer
if Observer == true
    At = transpose(A_platform);
    Ct = transpose(C_platform);

    sys = ss(A_platform,B_platform,C_platform,D_platform);

    % new_poles = 10*real(old_poles);
    % new_poles = [-10;-10.25;-20;-20.25;-7;-7.25;-14;-14.25;-8;-8.25;-5;-5.25;-18;-18.25;-7.8;-7.9;-3;-3.25;-3.5;-3.75];

    R = cov(noise);
    Q = cov(filtered);
    % load('Q.mat','Q')

    % nn = xcov([eta,Plant(:,2:end)]);
    % N = 0


    [~,L_platform,~] = kalman(sys,Q,R);
    % L_platform = diag(5*ones(1,20));

    eig(A_platform - L_platform*C_platform)

    % [K,prec] = place(At,Ct,new_poles)
    % L_platform = transpose(K);
else
    L_platform = zeros(width(A_platform),height(C_platform));
end
%% Run Simulink Model
cd('Models/Simulink');
SimulationOutput = sim(simulink_model,'StartTime','29.975',...
                        'StopTime',num2str(max(time)),'SrcWorkspace', 'current');
cd(home)
%% Extract Outputs & Zero-Mean
outputs = SimulationOutput.platform_out.Data;
simTime = SimulationOutput.tout;

%% Save Output Data Structure
cd(sprintf('Simulations\\%s',test));

outStruct.Time = simTime;
for i = 1:length(outputNames)
    outStruct.(outputNames{i}) = outputs(:,i);
end

if Observer == true
    slx_obs_results = outStruct;
    save('SimulinkObserver_Results','slx_obs_results');
else
    slx_results = outStruct;
    save('Simulink_Results',"slx_results");
end
cd(home)


