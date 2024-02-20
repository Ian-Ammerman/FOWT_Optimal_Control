function [nchan, nts, dt, chanNames] = readSIMO_resultstext(filename)

chanNames = [];
    nchan = 0;
    nts = 0;
    dt = 0;
fid = fopen(filename,'r');
if fid<0
    
    disp(['File ' filename ' could not open'])
    
else
    
    % read the header
    
    for ii = 1:6
        tline = fgetl(fid);
    end
    % number of samples
    tline = fgetl(fid);
    dat = textscan(tline,'%s %s %s %s %s %d');
    nts = dat{6};
    %  ** Time step  :    0.5000000   
    tline = fgetl(fid);
    dat = textscan(tline,'%s %s %s %s %f');
    dt = dat{5};
    % start time
    tline = fgetl(fid);
    dat = textscan(tline,'%s %s %s %f');
    tstart = dat{4};
    % end time
    tline = fgetl(fid);
    dat = textscan(tline,'%s %s %s %f');
    tend = dat{4}   ;
    
    
    tline = fgets(fid);
    while ischar(tline)
        if tline(1)=='*'
            % do nothing
        else
            nchan = nchan + 1;
            dat = textscan(tline,'%s');
            chanNames{nchan} = dat{1};
        end
        tline = fgets(fid);
    end
    fclose(fid);
end