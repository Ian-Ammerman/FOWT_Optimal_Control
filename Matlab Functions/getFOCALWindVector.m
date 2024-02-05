function wind = getFOCALWindVector(path,wind_case,time)

% Written By: Ian Ammerman
% Written: 1/31/24
%
% This helper function opens the correct wind velocity file for the FOCAL
% campaign 4 test campaign.
%
% Wind Cases:
% ------------
% 1 - W01 (rated > above rated > rated)
% 2 - W02 (rated wind speed)
% 3 - W03 (above rated wind speed)

% Load in appropriate wind file
if  wind_case == 0
    wind = zeros(size(time));
else
   switch wind_case
        case 1 % W01 - Below Rated
            filename = 'W01_fullScale_20230505.wnd';
        case 2 % W02 - Rated
            filename = 'W02_fullScale_R02_20230606.wnd';
        case 3 % W03 - Above Rated   
            filename = 'W03_fullScale_R02_20230613.wnd';
    end
    
    % Compile wind file path
    fullpath = sprintf('%s\\%s',path,filename);
    
    % Match wind speed to time vector
    raw_wind = readmatrix(fullpath,'FileType','text');
    wind = pchip(raw_wind(:,1),raw_wind(:,2),time); 
end