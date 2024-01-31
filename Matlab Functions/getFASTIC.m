function IC = getFASTIC(results)

fields = {'PtfmSurge','PtfmSway','PtfmHeave','PtfmRoll','PtfmPitch','PtfmYaw','TTDspFA','TTDspSS'};

for i = 1:length(fields)
    IC(i,1) = mean(results.(fields{i}));
end

end