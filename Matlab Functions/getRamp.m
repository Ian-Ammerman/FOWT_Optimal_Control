function s = getRamp(initial,final,num_steps)

% Written By: Ian Ammerman
% Written: 1/31/24
%
% s = getRamp(initial,final,num_steps) outputs a vector of num_steps rows
% and as many columns as present in initial & final (must have same number
% of columns) which increases linearly within each column from the initial
% value to the final value.

% Check inputs are correct
if size(initial) ~= size(final)
    error('Initial & final point vectors must be the same length!')
end

% Pre-allocate s
s = zeros(num_steps,size(initial,2));

% Form ramp vector
for i = 1:size(initial,2)
    s(:,i) = linspace(initial(i),final(i),num_steps);
end