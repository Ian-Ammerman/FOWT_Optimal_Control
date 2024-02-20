% readTimeDomainResults  Reads in results from a single simulation 
% 

close all
clear all
clc

%% User-defined input

folder = 'C:\Users\bachynsk\OneDrive - NTNU\Teaching\IntegratedDynamicAnalysisOfWindTurbines\2016\simaWorkspace\modWFloat\Initial\';
prefix = 'sima';
hullBody = 1; % typically 1 
numForChan = 123; % this is the number of channels in the force results ...
% storage. You can find this in the key_sima_elmfor.txt file. 

%% READ SIMULATION OUTPUT
%==========================================================================
%%READ SIMULATION OUTPUT
    % Read the SIMO results text file to get the channel names and number
    % of steps
    [nchan, nts, dt, chanNames] = readSIMO_resultstext([folder '\results.txt']);
    % Read the binary file
    AA = read_simoresults([folder '\results.tda'],nts);
    sizeAA = size(AA);
    if (sizeAA(1)<nts || nts<1); disp('Unable to read SIMO results'); return; end;
    % Determine which channels to read for the platform motions, wave
    % elevation
    [chanMotions, chanWave] = getchannelNumbers(chanNames,hullBody);
    if (chanMotions(1)<1 || chanWave<1); disp('Unable to read SIMO results'); return; end;
    time_SIMO = AA(:,2);
    
    % summarize data in matrix
    PlatMotions = AA(:,chanMotions);
    wave = AA(:,chanWave);

    % Read the wind turbine results 
    fname = [folder '\' prefix '_witurb.bin'];
    CC = read_rifbin(fname,0,26);
    sizeCC = size(CC);
    Nt = sizeCC(1);  % get the number of time steps
    if Nt<2; disp('Unable to read RIFLEX wind turbine results'); return; end; 
    
    time_WT = CC(:,2);
    omega = CC(:,3)*pi/180; % convert from deg/s to rad/s
    genTq = CC(:,5); 
    genPwr = CC(:,6); %
    azimuth = CC(:,7); 
    HubWindX = CC(:,8);
    HubWindY = CC(:,9);
    HubWindZ = CC(:,10);
    AeroForceX = CC(:,11);
    AeroMomX = CC(:,14);
    Bl1Pitch = CC(:,17);
    Bl2Pitch = CC(:,18);
    Bl3Pitch = CC(:,19); 

    % Read the RIFLEX force output file
    fname = [folder '\' prefix '_elmfor.bin'];
    BB = read_rifbin(fname,0,numForChan);
    sizeBB = size(BB);
    if(sizeBB(1)<1); disp('Unable to read RIFLEX force results'); return; end;  

    time_RIFLEX = BB(:,2); 
    % Tower Base
    TowerBaseAxial  = BB(:,3);
    TowerBaseTors   = BB(:,4);
    TowerBaseBMY    = BB(:,5);
    TowerBaseBMZ    = BB(:,7);
    TowerBaseShearY = BB(:,9);
    TowerBaseShearZ = BB(:,11);

    % Tower Top
    TowerTopAxial  = BB(:,13);
    TowerTopTors   = BB(:,14);
    TowerTopBMY    = BB(:,16); % end 2
    TowerTopBMZ    = BB(:,18);
    TowerTopShearY = BB(:,20);
    TowerTopShearZ = BB(:,22);    
    
    % Blade 1
    bl1Axial        = BB(:,23);
    bl1Tors         = BB(:,24);
    bl1BMY          = BB(:,25);
    bl1BMZ          = BB(:,27);
    bl1ShearY       = BB(:,29);
    bl1ShearZ       = BB(:,31);
    
    % Blade 2
    bl2Axial        = BB(:,33);
    bl2Tors         = BB(:,34);
    bl2BMY          = BB(:,35);
    bl2BMZ          = BB(:,37);
    bl2ShearY       = BB(:,39);
    bl2ShearZ       = BB(:,41);
    
    
    clear('fid','fname','data','BB','AA','CC','sizeBB','chanNames','chanWave','dt')
%%END READ SIMULATION OUTPUT

%==========================================================================

%% DESCRIPTION OF TIME SERIES available in the workspace
% =========================================================================
% Platform motions and wave elevation:
%   Time variable:             time_SIMO (mx1)  : s
%   Platform positions:        PlatMotions (mx6): m,m,m,deg,deg,deg
%   Wave elevation at origin   wave(mx1)        : m 

% Wind turbine results : 
%   Time variable:             time_RIFLEX (jx1): s
%   Thrust (shaft x):          AeroForceX (jx1) : kN
%   Aero. torque (about shaft):AeroMomX (jx1)   : kNm (rotor inertia NOT INCLUDED) 
%   Wind speed (shaft x):      HubWindX (jx1)   : m/s (at hub pos)
%   Wind speed (shaft y):      HubWindY (jx1)   : m/s (at hub pos)
%   Wind speed (shaft z):      HubWindZ (jx1)   : m/s (at hub pos)
%   Rotor speed:               omega (jx1)      : rad/s
%   Generator power:           genPwr (jx1)     : kW (including efficiency)
%   Generator torque:          genTq (jx1)      : kNm (HSS)
%   Azimuth angle:             azimuth (jx1)    : deg 

% Internal force results at the tower base: 
%   Time variable:              time_RIFLEX (jx1)       : s
%   Axial force at tower base:  TowerBaseAxial (jx1)    : kN  (DOF1)
%   Bending mom. Y at tow base: TowerBaseBMY (jx1)      : kNm (DOF3)
%   Bending mom. Z at tow base: TowerBaseBMZ (jx1)      : kNm (DOF5)
%   Shear force Y at tow base:  TowerBaseShearY (jx1)   : kN  (DOF7)
%   Shear force Z at tow base:  TowerBaseShearZ (jx1)   : kN  (DOF9)
%   Torsional moment tow base:  TowerBaseTors (jx1)     : kNm (DOF2)

% Internal force results at the blade root for blade 1: 
%   Time variable:              time_RIFLEX (jx1)       : s
%   Axial force at tower base:  bl1Axial (jx1)          : kN  (DOF1)
%   Bending mom. Y at tow base: bl1BMY (jx1)            : kNm (DOF3)
%   Bending mom. Z at tow base: bl1BMZ (jx1)            : kNm (DOF5)
%   Shear force Y at tow base:  bl1ShearY (jx1)         : kN  (DOF7)
%   Shear force Z at tow base:  bl1ShearZ (jx1)         : kN  (DOF9)
%   Torsional moment tow base:  bl1Tors (jx1)           : kNm (DOF2)

% Internal force results at the blade root for blade 2: 
%   Time variable:              time_RIFLEX (jx1)       : s
%   Axial force at tower base:  bl2Axial (jx1)          : kN  (DOF1)
%   Bending mom. Y at tow base: bl2BMY (jx1)            : kNm (DOF3)
%   Bending mom. Z at tow base: bl2BMZ (jx1)            : kNm (DOF5)
%   Shear force Y at tow base:  bl2ShearY (jx1)         : kN  (DOF7)
%   Shear force Z at tow base:  bl2ShearZ (jx1)         : kN  (DOF9)
%   Torsional moment tow base:  bl2Tors (jx1)           : kNm (DOF2)

% =========================================================================
%% Post-processing
%  Statistical analysis, fatigue calculation, etc
%  Remember to cut out the first part of the time series
%  Hint: 
%           t_start = 400;
%           indS = sum(time_SIMO<=t_start);
%           mean(PlatMotions(indS:end))



