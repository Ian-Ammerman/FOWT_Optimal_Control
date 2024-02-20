function D=cc2dam_2slope(cc,beta,K,stresslim,thk,tref,k)
% CC2DAM_2SLOPE  Calculates the total Palmgren-Miner damage of a cycle count.
% (two-slope SN curve)
% 
% CALL: D = cc2dam_2slope(cc,beta,K,stresslim,thk,tref,k);
% 
%   D    = Damage.                                                     [1]
%
%   cc   = Cycle count with minima in column 1 and                    [nx2]
%          maxima in column 2. (MPa)
%   beta = Beta-values, material parameter.                           [1x2]
%   K    = K-values, material parameter. (Optional, Default: 1)       [1x2]
%   stresslim = stress limit for first value of beta and K             [1]
%     (Optional, Default: 0)
%   thk  = cross-section thickness (m)  (Optional, Default: 0.025)      [1]
%   tref = reference thickness (m)      (Optional, Default: 0.025)      [1]      
%   k    = exponent for the thickness factor (Optional, Default: 0.0)  [1]
%
% The damage is calculated according to
%   D(i) = sum ( K * S^beta(i) ),  with  S = (max-min)/2
%   note that the factor K is modified to account for the DNV standard:
%       K = K*(2.0^beta)*(th/tref)^(k*beta)
%   
% History:
% Revised by PJ  01-Nov-1999
% - updated for WAFO
% Created by PJ (Pär Johannesson) 1997
%   from 'Toolbox: Rainflow Cycles for Switching Processes V.1.0'
% Modified by E Bachynski for 2-slope SN curve 2013


% Check input and otput
ni = nargin;
no = nargout;
% narginchk(2,7);

if ni < 3
  K=[];
  stresslim = 0;
  thk = 0.025;
  tref = 0.025;
  k = 0;
end

% Set default values
if isempty(K)
  K =  [1 1];
end

% Calculate damage

if (thk < tref)
    thk = tref;
end

% amplitudes
amp = abs(cc(:,2)-cc(:,1))/2; 

% sum over amplitudes
n=length(amp); 
Di=zeros(1,n);
D = 0;
for ii=1:n
    ampi = amp(ii);
    if ampi>stresslim/2 % few cycles 
        % note that the input stresslim is for stress ranges
        % here we are dealing with stress amplitudes
        betai = beta(1);
        fac1 = (2^(betai));
        fac2 = (thk/tref)^(k*betai);
        Ki  = K(1)*fac1*fac2;
    else % many cycles
        betai = beta(2);
        fac1 = (2^(betai));
        fac2 = (thk/tref)^(k*betai);
        Ki = K(2)*fac1*fac2;
    end
      D = D + Ki*(ampi^betai);
      Di(ii)=Ki*ampi^betai;

end

% D = sum(Di);
