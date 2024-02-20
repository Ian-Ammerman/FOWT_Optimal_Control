function [chanMotions, chanWave] = getchannelNumbers(chanNames,B1)
chanMotions = 0;
ind1 = 0;
chanWave = 0;
nameMotion = sprintf('B%dr29c1',B1);
ind1=0;
for ii = 1:length(chanNames)
   x = strfind(chanNames{ii}{1},nameMotion);
   if x==1
      ind1 = ii;
   end     
end
if ind1>0
    chanMotions = ind1 + [0:5];
end

ind1 = 0;
nameWave = 'Totalwaveelevation';
for ii = 1:length(chanNames)
   x = strfind(chanNames{ii}{1},nameWave);
   if x>0
      ind1 = ii;
   end     
end
chanWave = ind1;

