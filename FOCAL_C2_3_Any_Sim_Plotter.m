%% Plot SS Results Against Full Simulation Results
close all; clear all; clc;

% Define time offset if .ssexctn files used
tc = struct();
    tc.OpenFAST = 0;
    tc.Simulink = 29.975;
    tc.Experiment = 0;
    tc.Observer = 29.975;

%% --------------- User Inputs ---------------- %%
% Comparison flag (1: OpenFAST | 2: Simulink | 3: Experimental | 4: Kalman Filter)
type = [1];
plot_mark = {'none','none','none'};
xrange = [1,7500];
% xrange = [240 450];

% Descriptions
desc = {'OpenFAST';
        'StateSpace';
        'Experiment';
        'Kalman'};

color = {"#0072BD";
         "#D95319";
         [0.3,0.3,0.3];
         "#7E2F8E"};

style = {'-';
         '-';
         '--';
         '-'};

width = [1,1,1,1.25];

% vars = {'Wave1Elev'};
units = {'deg','m','RPM'};
% units = {'m','m','m','deg','deg','deg'};

% vars = {'PtfmPitch','TwrBsMyt','PtfmHeave'};
vars = {'PtfmPitch','PtfmSurge','GenSpeed'};
% vars = {'PtfmSurge','PtfmSway','PtfmHeave','PtfmRoll','PtfmPitch','PtfmYaw'}; %,'TTDspFA','TTDspSS','GenSpeed'}
% % vars = {'PtfmHeave','Wave1Elev'}
% % vars = {'TwrBsFxt','TwrBsFyt','TwrBsFzt','TwrBsMxt','TwrBsMyt','TwrBsMzt'}
% units = {'deg','N','m'}

for i = 1:length(type)
    t = type(i);

    switch t
        case 1 % OpenFAST Non-Linear Simulation Results
            load('OpenFAST_Results.mat');
            sim_results.Time = sim_results.Time-tc.OpenFAST; 
            sim_results = renameStructField(sim_results,'T_1_','FAIRTEN1');
            sim_results = renameStructField(sim_results,'T_2_','FAIRTEN2');
            sim_results = renameStructField(sim_results,'T_3_','FAIRTEN3');
            full_results{i} = sim_results;
            clear sim_results;

        case 2 % Simulink State Space Simulation
            load('Simulink_Results.mat');
            slx_results.Time = slx_results.Time-tc.Simulink;

            slx_results = renameStructField(slx_results,'T1','FAIRTEN1');
            slx_results = renameStructField(slx_results,'T2','FAIRTEN2');
            slx_results = renameStructField(slx_results,'T3','FAIRTEN3');

            full_results{i} = slx_results;
            clear slx_results;

        case 3 % Experimental Results
            load('Test_Results.mat')
            test_results.Time = test_results.Time - tc.Experiment;
            test_results = renameStructField(test_results,'T_1_','FAIRTEN1');
            test_results = renameStructField(test_results,'HSShftV','GenSpeed');
            test_results = renameStructField(test_results,'T_2_','FAIRTEN2');
            test_results = renameStructField(test_results,'T_3_','FAIRTEN3');
            test_results = renameStructField(test_results,'TwrBsAx','PtfmTAxt');
            test_results = renameStructField(test_results,'TwrBsAz','PtfmTAzt');

            % test_results.GenSpeed = test_results.GenSpeed*9.4;
            
            % test_results.TwrBsMxt = test_results.TwrBsMxt*10^-3;
            % test_results.TwrBsMyt = test_results.TwrBsMyt*10^-3;
            % test_results.TwrBsMzt = test_results.TwrBsMzt*10^-3;
            % test_results.TwrBsFxt = test_results.TwrBsFxt*10^-3;
            % test_results.TwrBsFyt = test_results.TwrBsFyt*10^-3;
            % test_results.TwrBsFzt = test_results.TwrBsFzt*10^-3;

            full_results{i} = test_results;
            clear test_results

        % case 4 % Simulink with Observer
        %     load('SimulinkObserver_Results.mat');
        %     slx_obs_results.Time = slx_obs_results.Time - tc.Observer
        % 
        %     slx_obs_results = renameStructField(slx_obs_results,'T1','FAIRTEN1');
        %     slx_obs_results = renameStructField(slx_obs_results,'T2','FAIRTEN2');
        %     slx_obs_results = renameStructField(slx_obs_results,'T3','FAIRTEN3');
        %     % slx_obs_results.TwrBsMyt = lowpass(slx_obs_results.TwrBsMyt,0.16,24);
        %     full_results{i} = slx_obs_results;
        %     clear slx_obs_results

        case 4
            load('Kalman_Results.mat','kalman_results');
            kalman_results.Time = kalman_results.Time-tc.Observer;

            full_results{i} = kalman_results;
            clear kalman_results;
    end
end


for i = 1:length(type)
    plotvar.(desc{type(i)}).Time = full_results{i}.Time;
    for j = 1:length(vars)
        try
            vals = full_results{i}.(vars{j});
            % vals = rMean(vals);
            % vals = vals - mean(vals(1:10));
            plotvar.(desc{type(i)}).(vars{j}) = vals;
        catch
            fprintf('Could not store %s: Result not present in %s results! \n',vars{j},desc{type(i)});
        end
    end
end

%% -------------- Do Plotting --------------- %%
for i = 1:length(vars)
    figure('Name',sprintf('Model Comparison | %s',vars{i}))
    ax = gca; box on; hold on;
    set(ax,'FontSize',13)
    xlim(xrange)

    xlabel('Time [s]');
    try 
        if contains(vars{i},'Ptfm')
            y_axis = extractAfter(vars{i}, "Ptfm");
        else
            y_axis = vars{i};
        end
        ylabel(sprintf('%s [%s]',y_axis,units{i}));
    end

    for j = 1:length(type)
        try
            plot(plotvar.(desc{type(j)}).Time,plotvar.(desc{type(j)}).(vars{i}),...
                'DisplayName',desc{type(j)},'Color',color{type(j)},'LineStyle',style{type(j)},...
                'LineWidth',width(type(j)));
        catch
            fprintf('Could not plot %s: Result not present in %s results! \n',vars{i},desc{type(j)});
        end
    end

    legend('Location','best')
end







% figure('Name','SS to Simulation Comparison')
% num_plot = 3;
% for k = 1:num_plot
%     subplot(num_plot,1,k)
%     ax = gca; box on; hold on;
%     xlabel('Time [s]')
%     xlim(xrange);
%     title(varnames{k})
% 
%     switch k
%         case 1
%            for i = 1:length(type)
%                plot(time{i},var1{i},'DisplayName',desc{i});
%            end
% 
%         case 2
%             for i = 1:length(type)
%                 plot(time{i},var2{i},'DisplayName',desc{i});
%             end
% 
%         case 3
%             for i = 1:length(type)
%                 plot(time{i},var3{i},'DisplayName',desc{i});
%             end
%         
%         case 4
%             for i = 1:length(type)
%                 plot(time{i},var4{i},'DisplayName',desc{i});
%             end
%     end
% 
%     legend
% end