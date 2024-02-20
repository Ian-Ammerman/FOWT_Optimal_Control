function [A_platform,B_platform,C_platform,D_platform,x_OP,y_OP] = loadPlatformModel(platform_dir,state_range,control_range,dt)

% Load in raw files
load(sprintf('%s\\FOCAL_C4_A.mat',platform_dir),'A');
load(sprintf('%s\\FOCAL_C4_B.mat',platform_dir),'B');
load(sprintf('%s\\FOCAL_C4_C.mat',platform_dir),'C');
load(sprintf('%s\\FOCAL_C4_D.mat',platform_dir),'D');
load(sprintf('%s\\FOCAL_C4_X_OP.mat',platform_dir));
load(sprintf('%s\\FOCAL_C4_Y_OP.mat',platform_dir));

% Adjust model states & inputs
x_OP = x_OP(state_range);

A = A(state_range,state_range);
B = B(state_range,control_range);
C = C(:,state_range);
D = 0*D(:,control_range);

% Discretize Platform
platform_sys_c = ss(A,B,C,D);
platform_sys_d = c2d(platform_sys_c,dt,'zoh');

% Reduce model order
% % new_order = input('Reduced model order: ');
% new_order = 125;
% R = reducespec(platform_sys_d,'balanced');
% rsys = getrom(R,Order=new_order);

[A_platform,B_platform,C_platform,D_platform] = ssdata(platform_sys_d);

end