function getFOCALFD(filename)

% Create structure of data
for i = 1:length(labels)
    Test_Results.(labels{i}) = channels(:,i);
end

% Rename fields to match FAST
oldNames = {'Surge','Sway','Heave','Roll','Pitch','Yaw'};
newNames = {'PtfmSurge','PtfmSway','PtfmHeave','PtfmRoll','PtfmPitch','PtfmYaw'};

for i = 1:length(oldNames)
    try
        Test_Results = renameStructField(Test_Results,oldNames{i},newNames{i});
    catch
        fprintf('Could not find %! \n',oldNames{i});
    end
end

end