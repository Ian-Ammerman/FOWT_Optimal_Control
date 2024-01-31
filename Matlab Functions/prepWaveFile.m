function prepWaveFile(buffer,simulation)

%
% Written By: Ian Ammerman
% Written: 9/1/23
%
% prepWaveFile generates a .Elev file for use with OpenFAST from basin test
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
fs = length(time)/max(time);

try
    eta = lowpass(test_results.Wave1Elev,0.35,fs);
catch
    eta = zeros(size(test_results.Time));
    disp('Error in filtering wave elevation - setting elevation to zero.')
end

wave = [time,eta];

if buffer > 0
    dt = max(time)/length(time);
    ttime = [0:dt:max(time)+buffer]';
    teta = [zeros(length(ttime)-length(time),1);
           eta];

    wave = [ttime,teta];
end

% Save in Wave_Files
cd('..\..')
writematrix(wave,'Wave_Files\InputWave.Elev','FileType','text','Delimiter','\t');

end