function prepWindFile(buffer,simulation)

%
% Written By: Ian Ammerman
% Written: 9/1/23
%
% prepWindFile generates a .dat uniform wind field file for use with OpenFAST from basin test
% results
%
% Inputs:
% -----------
% buffer - amount of settling time to add to the start
% simulation - test folder with results

cd(simulation);
try
    load('Test_Results.mat','test_results');
catch
    disp('No test file found.')
    return
end

time = test_results.Time;

try
    u = test_results.Wind1VelX;
catch
    u = zeros(size(test_results.Time));
end

if buffer > 0
    dt = max(time)/length(time);
    ttime = [0:dt:max(time)+buffer]';
    unew = [zeros(length(ttime)-length(time),1); u];
    
    wind = [ttime,unew,zeros(height(ttime),6)];
else
    wind = [time,u,zeros(height(time),6)];
end

% Zero out negative wind
wind(wind<0) = 0;

cd('..\..')

% Save in Wave_Files
writematrix(wind,'Wind_Files/InputWind.dat','Delimiter','\t');



end