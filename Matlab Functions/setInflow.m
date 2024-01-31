function setInflow(filepath,line,windspeed)

%% Open File
file_lines = readlines(filepath,'WhitespaceRule','trimleading');

%% Define Line Number
% line_number = 13; % FOCAL C4 Dev Version
line_number = line; % v3.5.0 & v3.5.1

%% Change Inflow Wind Speed Value
% Isolate & separate line
active_line = split(file_lines{line_number});

% Set windspeed value
active_line{1} = num2str(windspeed);

% Recombine line to full set
file_lines{line_number} = char(join(active_line,'   '));

%% Re-save New File
writelines(file_lines,filepath);