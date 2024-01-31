function [IC,lines,original] = setED(filepath,varargin)


%% Parse Inputs Values
p = inputParser
addRequired(p,'filepath')
addParameter(p,'PtfmSurge',NaN);
addParameter(p,'PtfmSway',NaN);
addParameter(p,'PtfmHeave',NaN);
addParameter(p,'PtfmRoll',NaN);
addParameter(p,'PtfmPitch',NaN);
addParameter(p,'PtfmYaw',NaN);
addParameter(p,'TTDspFA',NaN);
addParameter(p,'TTDspSS',NaN);
parse(p,filepath,varargin{:})

IC = [[26:42]',zeros(17,1)];

IC(10,2) = p.Results.TTDspFA;
IC(11,2) = p.Results.TTDspSS;
IC(12,2) = p.Results.PtfmSurge;
IC(13,2) = p.Results.PtfmSway;
IC(14,2) = p.Results.PtfmHeave;
IC(15,2) = p.Results.PtfmRoll;
IC(16,2) = p.Results.PtfmPitch;
IC(17,2) = p.Results.PtfmYaw;

%% Load in ElastoDyn File
lines = readlines(filepath,'WhitespaceRule','trimleading');
original = lines;

%% Loop Over IC Vector and Alter Appropriate Lines
for i = 1:size(IC,1)
    if ~isnan(IC(i,2))
        line = split(lines{IC(i,1)});
        line{1} = num2str(IC(i,2));
        lines{IC(i,1)} = char(join(line,'   '));
    end
end

%% Save File in Same Place
writelines(lines,filepath)