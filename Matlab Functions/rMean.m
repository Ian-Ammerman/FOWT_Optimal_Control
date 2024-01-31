function newdata = rMean(data)

% Written By: Ian Ammerman
%
% rMean removes the mean value of time series data.
%
% Inputs:
% ----------
% data - time series data
%
% Output:
% ----------
% newdata - data with mean value removed

%% -------------------- Begin Code ------------------- %%

newdata = data-mean(data);

end

