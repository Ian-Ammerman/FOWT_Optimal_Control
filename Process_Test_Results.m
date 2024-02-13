% Script to format test results from FOCAL Campaign 4


%%% USER MUST OPEN TEST FILE BEFORE RUNNING SCRIPT

%% Form Structure
field_names = labels;

for i = 1:length(field_names)
    test_results.(field_names{i}) = channels(:,i);
end

%% Rename Key Fields to Match OpenFAST
old_names = {'waveStaff5','Surge','Sway','Heave','Roll','Pitch','Yaw'};
new_names = {'Wave1Elev','PtfmSurge','PtfmSway','PtfmHeave','PtfmRoll','PtfmPitch','PtfmYaw'};

for j = 1:length(old_names)
    test_results = renameStructField(test_results,old_names{j},new_names{j});
end