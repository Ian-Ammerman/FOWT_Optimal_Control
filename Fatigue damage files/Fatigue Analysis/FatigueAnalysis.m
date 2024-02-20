function [FatAx] = FatigueAnalysis(OutSummary)

% Compute the fatigue damage on a hollow circular section for given loads
% Modified by Amrit Verma 02 Feb 24


% Inputs:
%%%%%%%% Cross section geometry
% r1: inner radius (m)                                                      [1]
% r2: outer radius (m)                                                      [1]

%%%%%%%% Loads time history
% t:  time vector (s)                                                       [nx1]
% note: n = length(t)
% Nx: axial load (kN)                                                       [nx1]
% My: bending moment about local y (kNm)                                    [nx1]
% Mz: bending moment about local z (kNm)                                    [nx1]
% Mx: torisonal load (kNm)                                                  [nx1]
% Vy: shear force in y (kN)                                                 [nx1]
% Vz: shear force in z (kN)                                                 [nx1]
% tstart: start time for damage calculation (s)                             [1]
% tend: end time for damage calculation (s)                                 [1]

%%%%%%%% Parameters for axial stress calculation
% beta_sx: slopes for S-N curves, axial stress                              [1x2]
% K_sx: [1/(10^log(a1)) 1/(10^log(a2))] for axial stress                    [1x2]
% stresslim_sx: minimum stress (MPa) for curve 1, axial stress              [1]
% note: stresslim comes directly from DNV curves (ie, input for range)
% tref_sx: reference thickness for axial stress (m)                         [1]
% k_sx: exponent for thickness factor for axial stress                      [1]

%%%%%%%% Parameters for shear stress calculation
% beta_txy: slopes for S-N curves, shear stress                             [1x2]
% K_txy: [1/(10^log(a1)) 1/(10^log(a2))] for shear stress                   [1x2]
% stresslim_txy: minimum stress (MPa) for curve 1, shear stress             [1]
% note: stresslim comes directly from DNV curves (ie, input for range)
% tref_txy: reference thickness for shear stress (m)                        [1]
% k_txy: exponent for thickness factor for shear stress                     [1]


%%%%%%%% Stress calculation parameters
% locAmp: location for output of amplitude bins (integer 1-nlocs)           [1]
% nlocs: number of locations about the cross section for damage calc        [1]
% bpass_upper: upper frequency limit (Hz) for bandpass of stress signal
% prior to damage calculation. no bandpass filter for bpass_upper<0.        [1]

%%%%%%%%
% OUTPUTS
%%%%%%%%
% FatAx.DRFC_sx: summed fatigue damage due to axial stress (nlocs,1)
% FatAx.DRFC_txy: summed fatigue damage due to shear stress (nlocs,1)
% FatAx.Stat.ms: mean axial stress (:,1) and shear stress (:,2) [nlocs,2] (MPa)
% FatAx.Stats.ts: standard deviation of the axial stress (:,1) and shear stress (:,2)
% [nlocs,2]
% FatAx.Stat.ks: kurtosis of the axial stress (:,1) and shear stress (:,2) [nlocs,2]
% FatAx.Stat.sks: skewness of the axial stress (:,1) and shear stress (:,2) [nlocs,2]
% FatAx.stresses_Sx: axial stress time history [n,nlocs]
% FatAx.stresses_Txy: shear stress time history [n,nlocs]
% ampRFC: Sx amplitude bins output for location locAmp [N,1]
% N is not known a priori

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nlocs =36; 
bpass_upper=-1;
ShearDamage = 'no';

% Taking results from Riflex
t   = OutSummary.bin_Results.Element_Forces(1).Data.Time          ;
Nx  = OutSummary.bin_Results.Element_Forces(1).Data.Glob_Fax    ;
Vy  = OutSummary.bin_Results.Element_Forces(1).Data.Glob_Qy2    ;
Vz  = OutSummary.bin_Results.Element_Forces(1).Data.Glob_Qx2    ;
Mx  = OutSummary.bin_Results.Element_Forces(1).Data.Glob_Tr     ;
My  = OutSummary.bin_Results.Element_Forces(1).Data.Glob_My2    ;
Mz  = OutSummary.bin_Results.Element_Forces(1).Data.Glob_Mx2    ;
        

% Properties of the cross section
r1  =  	OutSummary.Fatigue.Tower_IR(1);
r2  =	OutSummary.Fatigue.Tower_OR(1);
thk =  	OutSummary.Fatigue.Tower_Thc(1);
A_base = pi*(r2^2-r1^2);        % Area
I_base = pi/4*(r2^4-r1^4);      % Area moment of inertia
J_base = pi/2*(r2^4-r1^4);      % Polar moment of area

% Time interval for stress and fatigue analysis
tstart  =   OutSummary.InputParam.Time_Increment;
tend    =   OutSummary.InputParam.SimulationTime-OutSummary.InputParam.cut_off_time;

s_ind = sum(t<=tstart)+1;
e_ind = sum(t<=tend);

% Initialize outputs
locs                    = 1:nlocs+1;

FatAx.DRFC_sx           = zeros(nlocs,1);
FatAx.Stat.ms           = zeros(nlocs,1);
FatAx.Stat.stnd         = zeros(nlocs,1);
FatAx.Stat.ks           = zeros(nlocs,1);
FatAx.Stat.sks          = zeros(nlocs,1);
FatAx.stresses_Sx       = zeros(length(t),nlocs);

FatSh.DRFC_txy      	= zeros(nlocs,1);
FatSh.Stat.ms           = zeros(nlocs,1);
FatSh.Stat.stnd         = zeros(nlocs,1);
FatSh.Stat.ks           = zeros(nlocs,1);
FatSh.Stat.sks          = zeros(nlocs,1);
FatSh.stresses_Txy      = zeros(length(t),nlocs);

% Negative slopes of S-N curve on logN-logS plot
beta_sx         = [OutSummary.Fatigue.m1_ax  OutSummary.Fatigue.m2_ax];
beta_txy        = [OutSummary.Fatigue.m1_sh OutSummary.Fatigue.m2_sh];
K_sx            = [1/(10^OutSummary.Fatigue.loga1_ax)  1/(10^OutSummary.Fatigue.loga2_ax)];
K_txy           = [1/(10^OutSummary.Fatigue.loga1_sh)  1/(10^OutSummary.Fatigue.loga2_sh)];
stresslim_sx    = OutSummary.Fatigue.slim_ax_tower;
stresslim_txy   = OutSummary.Fatigue.slim_sh;


% Reference Thickness
tref_sx         = OutSummary.Fatigue.Tref_ax;
tref_txy        = OutSummary.Fatigue.Tref_sh;

% Thickness Exponent
k_sx            = OutSummary.Fatigue.k_ax;
k_txy           = OutSummary.Fatigue.k_sh;

FatAx.Angle=[0:360/nlocs:360]'; %#ok<NBRAK>
FatSh.Angle=[0:360/nlocs:360]'; %#ok<NBRAK>

% Loop through the points around the cross section
for cs_points = 1:nlocs+1
    loc = locs(cs_points);
    theta = 2*pi/nlocs*(loc-1);
    
    % ------------------ FATIGUE DAMAGE DUE TO AXIAL STRESS ----------------- %
    % ------------------ FATIGUE DAMAGE DUE TO AXIAL STRESS ----------------- %
    
    % AXIAL STRESS compute stress from loads at this location
    stress1     = (-My*r2*cos(theta)/I_base + Mz*r2*sin(theta)/I_base + Nx/A_base)/1000; % MPa
    
    % Band-pass filter axial stress
    if bpass_upper > 0
        stressSx            = bpass(stress1,(t(2)-t(1)),0.0,bpass_upper);
    else
        stressSx = stress1;
    end
    
    % Store stress for output
    FatAx.stresses_Sx(:,cs_points)  = stressSx;
    
    % Statistics of the axial stress signal at this point (only for selected time!)
    FatAx.Stat.ms(cs_points,1)      = mean(stressSx(s_ind:e_ind));
    FatAx.Stat.stnd(cs_points,1)    = std(stressSx(s_ind:e_ind));
    FatAx.Stat.ks(cs_points,1)      = kurtosis(stressSx(s_ind:e_ind));
    FatAx.Stat.sks(cs_points,1)     = skewness(stressSx(s_ind:e_ind));
    
    x = [t(s_ind:e_ind) stressSx(s_ind:e_ind)];
    x(:,1) = x(:,1);
    
    % Turning points
    [tp, ~] = dat2tp(x);
    def.res = 'CS';
    def.asymmetric = 0;
    def.time = 0;
    
    % Rainflow cycles
    [RFC,~,~,def] = tp2rfc(tp,def);
    
    FatAx_RFC(cs_points).cc  	= RFC;
    FatAx.DRFC_sx(cs_points)    = E2_cc2dam_2slope(RFC,beta_sx,K_sx,stresslim_sx,thk,tref_sx,k_sx);
    
    % ---------------- FATIGUE DAMAGE DUE TO SHEAR STRESS --------------- %
    % ---------------- FATIGUE DAMAGE DUE TO SHEAR STRESS --------------- %
    
    switch(lower(ShearDamage))
        case {'yes'}
            
            % SHEAR STRESS - Compute stress from loads at this location
            stress2     = (Mx*r2/J_base + 2*Vy/A_base*sin(theta) + 2*Vz/A_base*cos(theta))/1000; % MPa
            
            % Band-pass filter shear stress
            bpass_upper=-1
            if bpass_upper > 0
                stressT             = bpass(stress2,(t(2)-t(1)),0.0,bpass_upper);
            else
                stressT = stress2;
            end
            
            % Store stress for output
            FatSh.stresses_Txy(:,cs_points) = stressT;
            
            % Statistics of the shear stress signal at this point (only for selected time!)
            FatSh.Stat.ms(cs_points,1)      = mean(stressT(s_ind:e_ind));
            FatSh.Stat.stnd(cs_points,1)    = std(stressT(s_ind:e_ind));
            FatSh.Stat.ks(cs_points,1)      = kurtosis(stressT(s_ind:e_ind));
            FatSh.Stat.sks(cs_points,1)     = skewness(stressT(s_ind:e_ind));
            
            x = [t(s_ind:e_ind) stressT(s_ind:e_ind)];
            x(:,1) = x(:,1);
            
            % Turning points
            [tp, ~] = dat2tp(x);
            
            % Rainflow cycles
            [RFC,~,~,def] = tp2rfc(tp,def);
            
            FatSh.DRFC_txy(cs_points) = E2_cc2dam_2slope(RFC,beta_txy,K_txy,stresslim_txy,thk,tref_txy,k_txy);
            
        otherwise
            FatSh='Not Calculated';
    end
end
