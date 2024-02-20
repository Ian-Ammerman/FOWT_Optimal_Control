close all
clear all
clc


% generate a wind file for step wind. The file format is: 

%File head: (The file head may contain an arbitrary number of comment lines starting with:‘)
%Line 1 : Number of samples
%Line 2 : Time step
%Line 3 : Number of columns (=2)
%Time series:
%Line 4 : velocity direction
%Line 5-N : velocity direction


% The obtained file can be used with the "fluctuating two-component" wind
% option in SIMA. 

% inputs: 
fname = 'step_wind_long_return.asc'; % output file
WS = [3:1:25 24:-1:3];  
dt = 0.1;
tWS = 200; % duration for each wind speed
tstart = 600; % extra time for first wind speed

% Compute the length of the file
nWS = length(WS); 
tTot = tWS*nWS + tstart; 

% generate time and wind vectors
t = (0:dt:tTot).';
wvel = ones(length(t),1); 
nstart = sum(t<tstart); 
wvel(1:nstart) = WS(1)*wvel(1:nstart); 
ind1 = nstart+1; 
npWS = tWS/dt; 

for ii = 1:length(WS)
    wvel(ind1:ind1+npWS) = WS(ii); 
    ind1 = ind1 + npWS; 
end

% Plot the time series to check
figure
plot(t,wvel,'k')
xlabel('Time, s')
ylabel('V_h_u_b, m/s')
grid on
ylim([0 25])

% Write the time series to file
fid = fopen(fname,'w');
fprintf(fid,'%d\n',length(t));
fprintf(fid,'%f\n',dt); 
fprintf(fid,'%d\n',2); 
for ii = 1:length(wvel)
    fprintf(fid,'%f %f  \n',wvel(ii),0);
end
fclose(fid); 
