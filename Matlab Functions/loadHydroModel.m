function [A_hydro,B_hydro,C_hydro,D_hydro] = loadHydroModel(filedir,dt)

% Load in raw files
load(sprintf('%s\\FOCAL_C4_HD_A.mat',filedir),'A');
load(sprintf('%s\\FOCAL_C4_HD_B.mat',filedir),'B');
load(sprintf('%s\\FOCAL_C4_HD_C.mat',filedir),'C');
load(sprintf('%s\\FOCAL_C4_HD_D.mat',filedir),'D');
load(sprintf('%s\\FOCAL_C4_Hydro_OP.mat',filedir),'Hydro_OP');

% Convert Hydro_OP type
Hydro_OP = cell2mat(Hydro_OP);

% Trim inputs
B = B(:,[37,7,8,9,10,11,12]);
D = D(:,[37,7,8,9,10,11,12]);

% Discretize hydrodynamics model
hydro_sys_c = ss(A,B,C,D);
hydro_sys_d = c2d(hydro_sys_c,dt,'zoh');
[A_hydro,B_hydro,C_hydro,D_hydro] = ssdata(hydro_sys_d);

end