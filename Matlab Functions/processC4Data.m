function test_results = processC4Data(filename)
load(filename,'channels','labels','units');




test_results.units = [labels;units];
for i = 1:length(labels)
    test_results.(labels{i}) = channels(:,i);
end
