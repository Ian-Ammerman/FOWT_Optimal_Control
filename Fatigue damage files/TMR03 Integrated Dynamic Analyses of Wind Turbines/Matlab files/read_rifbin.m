function BB = read_rifbin(forfilebin,Nt,Nchan)
% read RIFLEX binary file
% forfilebin is the name of the binary file
% Nt is the number of time steps (if 0, then Nchan is needed)
% Nchan is an optional argument - unused if Nt>0

if nargin < 3
   Nchan = [];
end

BB = [];
iexistbin  = exist([forfilebin]);
  if (iexistbin)
    fid      = fopen([forfilebin],'rb');
    AA       = fread(fid,'single');
    fclose(fid);
    
    % Arrange data:
    if Nt>0 % calculate Nchan, otherwise Nchan is given 
        Nchan = max(size(AA))/Nt;
    end
    for ichan = 1:Nchan
      BB(:,ichan) = AA(ichan:Nchan:end)';
    end
    disp([' ... Binary file read: ' forfilebin ])
  end