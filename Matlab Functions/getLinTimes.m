function linTimes = getLinTimes(rpm,N,t0)

% Computes vector of linearization times given N steps per rotation,
% steady state rotor speed in rpm, and start time t0.

%% Convert to Rads/s
omega = rpm*(pi/30);

%% Compute Time per Rotation
rot_time = (2*pi)/omega;

%% Compute step time
step_time = rot_time/N;

%% Create Row Vector of Evenly Spaced Times
linTimes = linspace(t0,t0+rot_time,N);