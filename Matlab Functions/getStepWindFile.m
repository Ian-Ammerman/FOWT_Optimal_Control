function wind = getStepWindFile(num_steps,step_size,step_length)

% Written By: Ian Ammerman
% Last Modified: 1/9/24
%
% Generates a .wnd file for used with OpenFAST with a step wind case.
%
% Inputs:
% ----------
% num_steps - number of step increments in file
% step_size - change in wind speed between steps
% step_length - length (in seconds) of each step
%
% Outputs:
% ----------
% Step_Wind_N###_U###_T###.wnd - step wind file of .wnd extension

%% ------------------- BEGIN CODE ------------------ %%

% Initialize wind vector
dt = 0.05;
wind = zeros(num_steps*step_length*(1/dt),2);

% Fill out wind values
for i = 0:num_steps-1
    wind(step_length*i*(1/dt)+1:step_length*(i+1)*(1/dt),2) = step_size*(i+1);
end

% Time Values 
wind(:,1) = [dt:dt:dt*size(wind,1)]';

% Pad with zeros for OpenFAST format
wind = [wind,zeros(size(wind,1),6)];

% Save wind file
writematrix(wind,sprintf('Step_Wind_N%i_U%0.3g_T%0.3g.wnd',num_steps,step_size,step_length),'FileType','text');