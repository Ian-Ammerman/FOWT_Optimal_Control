function [results,units] = readFastTabular(filepath)

% Written By: Ian Ammerman
% Written: 5/31/23
% Last Modified: 5/31/23
% 
% readFastTabular serves to read the tab delimited output of OPENFast and
% return a structure consisting of the output variables and their units.
%
% Currently configures for OPENFast version: 3.1.0
%
% Inputs:
% --------
% filepath - full file path of target FAST output file.
%
% Outputs:
% --------
% results - structure containing each variable time series from FAST.
% units - cell array of units corrosponding to the variables in results.

% ------------------------------- CODE -------------------------------- %

% Define file ID from target file
fileID = fopen(filepath);

% Skip first 6 lines and read variable names & units
timestring = 'time';
varRow = false;
while varRow == false
    % Grab row
    row = fgetl(fileID);
    % Check if row of variable names
    if strncmpi(timestring,row,4) == 1
        % Store line length
        n = length(strsplit(row));
        % Save variable names
        varNames = strsplit(row);
        % Save variable units from next row
        varUnits = strsplit(fgetl(fileID));
        % Break while loop
        varRow = true;
    end
end

% Read numeric data
data = textscan(fileID,repmat('%f',1,n),'Delimiter','\t','EndOfLine','\r\n');

% Convert tension variable to string (MAP++)
for i = 1:length(varNames)
    string = varNames{i};
    new_string = strrep(string,'[','_');
    new_string = strrep(new_string,']','_');
    varNames{i} = new_string;
end

% Output final structure
results = cell2struct(data,varNames,2);
units = varUnits;

end