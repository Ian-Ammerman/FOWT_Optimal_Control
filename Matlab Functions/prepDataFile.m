function prepDataFile(filename)

load(filename,'channels','labels','units');

for i = 1:length(labels)
    test_results.(labels{i}) = channels(:,i);
end

% Rename structure fields
old = {'time','Surge','Sway','Heave','Roll','Pitch','Yaw','waveElev'};
new = {'Time','PtfmSurge','PtfmSway','PtfmHeave','PtfmRoll','PtfmPitch','PtfmYaw','Wave1Elev'};
for i = 1:length(old)
    test_results = renameStructField(test_results,old{i},new{i});
end

if ~isfield(test_results,'Wind1VelX')
    test_results.Wind1VelX = zeros(length(test_results.Time),1);
end

if ~isfield(test_results,'Wave1Elev')
    test_results.Wave1Elev = zeros(length(test_results.Time),1);
end

save('Test_Results.mat','test_results');

end