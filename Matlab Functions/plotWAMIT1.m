function plotWAMIT1(filename)

% Written By: Ian Ammerman
% Written: 2/15/24
%
% plotWAMIT1 plots the frequency dependent added mass and damping from the
% WAMIT .1 result file.
%
% Inputs:
% ---------
% filename - relative or absolute filepath to WAMIT file


%% Read in WAMIT File
wam = readmatrix(filename,'FileType','text');

%% Isolate Frequencies
period = unique(wam(:,1));
frequency = 1./period;

%% Separate DOFs
DOFs = {'Surge','Sway','Heave','Roll','Pitch','Yaw'};

for i = 1:6
    wam_sub = wam(wam(:,2)==i,[3,4,5]);
    wam_sub = wam_sub(wam_sub(:,1)==i,:);

    mass.(DOFs{i}) = wam_sub(:,2);
    damp.(DOFs{i}) = wam_sub(:,3);
end

%% Plot Results
for i = 1:length(DOFs)

    % Create extra wide figure
    figure('Position', [400, 400, 1120, 420]); % [left bottom width height]

    % Plot added mass
    subplot(1,2,1)
    gca; box on;
    plot(frequency,mass.(DOFs{i}))
    title(sprintf('%s Added Mass [Dimensionless]',DOFs{i}))
    xlabel('Frequency [Hz]')

    % Plot damping
    subplot(1,2,2)
    gca; box on;
    plot(frequency,damp.(DOFs{i}))
    title(sprintf('%s Hydro Damping [Dimensionless]',DOFs{i}))
    xlabel('Frequency [Hz]')
end
