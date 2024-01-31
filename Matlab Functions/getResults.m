function getResults(variable)

% Load in Test & Simulation Results
load('Test_Results.mat','test_results');
load('Simulink_Results','slx_results');
load('OpenFAST_Results.mat','sim_results')
load('SimulinkObserver_Results','slx_obs_results');

% variable = 'PtfmHeave';
tc_sim = 0;
tc_slx = 29.975;
tc_obs = 29.975;

time = test_results.Time;
sim_time = sim_results.Time-tc_sim;
slx_time = slx_results.Time-tc_slx;
obs_time = slx_obs_results.Time-tc_obs;

% Determine end time of analysis
tstart = 0;
tstop = 4000; %min(max(sim_time),max(slx_time));

start_index = dsearchn(time,tstart);
stop_index = dsearchn(time,tstop);

time = time(start_index:stop_index);

% Sync data in time
ytest = rMean(test_results.(variable));
    ytest = ytest(start_index:stop_index);

yslx = pchip(slx_time,rMean(slx_results.(variable)),time);
ysim = pchip(sim_time,rMean(sim_results.(variable)),time);
yobs = pchip(obs_time,rMean(slx_obs_results.(variable)),time);

% Form matrix
ymat = [ytest,ysim,yslx,yobs];
[R,P,RLO,RUP] = testCorrelation(ymat,'alpha',0.00000001);

figure
subplot(2,1,1)
gca; hold on; box on;
plot(time,ytest,'DisplayName','Experiment')
plot(time,ysim,'DisplayName','OpenFAST')
plot(time,yslx,'DisplayName','State-Space')
plot(time,yobs,'DisplayName','Kalman Filter')
legend

Names = categorical(["Experiment","OpenFAST","State-Space","Kalman Filter"]);

subplot(2,1,2)
bar(["Experiment","OpenFAST","State-Space","Kalman Filter"],R,0.4); hold on;
errorbar(Names,R,R-RLO,RUP-R,'.');

%% Envelope Comparison
[test_upper,test_lower] = envelope(ytest,368,'rms');
[sim_upper,sim_lower] = envelope(ysim,368,'rms');
[slx_upper,slx_lower] = envelope(yslx,368,'rms');
[slx_obs_upper,slx_obs_lower] = envelope(yobs,368,'rms');

% Envelope correlation
ycorrmat = [test_upper,sim_upper,slx_upper,slx_obs_upper];
[Renv,Penv,RLOenv,RUPenv] = testCorrelation(ycorrmat);

figure
subplot(2,1,1)
gca; hold on; box on;
plot(time,test_upper,'DisplayName','Experiment');
plot(time,sim_upper,'DisplayName','OpenFAST');
plot(time,slx_upper,'DisplayName','State Space');
plot(time,slx_obs_upper,'DisplayName','Observer');
legend

title(sprintf('Upper Response Envelope | %s',variable))
% xlim([275 14000])
xlabel('Time [s]')
ylabel('N-m')

% Envelope Error
slx_env_abs = 100*abs(slx_upper - test_upper)./max(test_upper);
sim_env_abs = 100*abs(sim_upper - test_upper)./max(test_upper);
obs_env_abs = 100*abs(slx_obs_upper - test_upper)./max(test_upper);

subplot(2,1,2)
gca; hold on; box on; grid on;
plot(time,slx_env_abs,'DisplayName','State-Space')
plot(time,sim_env_abs,'DisplayName','OpenFAST')
plot(time,obs_env_abs,'DisplayName','Observer')

title(sprintf('Upper Envelope Percent Error | %s',variable));
% xlim([275 14000])
xlabel('Time [s]')
legend

%% PSD Comparison
% Compute PSDs
Fs = round(length(time)/max(time));
nsmooth = 15;
sim_psd = myPSD(ysim,Fs,nsmooth);
slx_psd = myPSD(yslx,Fs,nsmooth);
test_psd = myPSD(ytest,Fs,nsmooth);
obs_psd = myPSD(yobs,Fs,nsmooth);
freq = test_psd(:,1);

figure
subplot(2,1,1)
gca; hold on;
plot(sim_psd(:,1),sim_psd(:,2),'DisplayName','OpenFAST PSD');
plot(slx_psd(:,1),slx_psd(:,2),'DisplayName','StateSpace PSD');
plot(test_psd(:,1),test_psd(:,2),'DisplayName','Experimental PSD');
plot(obs_psd(:,1),obs_psd(:,2),'DisplayName','Observer PSD');

xlabel('Frequency [Hz]')
xlim([0.06 0.16])
title(sprintf('PSD | %s',variable));
legend

% Compute Absolute Error in PSDs
slx_abs = 100*abs(slx_psd-test_psd)./max(test_psd(:,2));
obs_abs = 100*abs(obs_psd-test_psd)./max(test_psd(:,2));
sim_abs = 100*abs(sim_psd-test_psd)./max(test_psd(:,2));

subplot(2,1,2)
gca; hold on;
plot(freq,slx_abs(:,2),'DisplayName','StateSpace');
plot(freq,sim_abs(:,2),'DisplayName','OpenFAST');
plot(freq,obs_abs(:,2),'DisplayName','Observer');

xlabel('Frequency [Hz]')
ylabel('% of Peak PSD Value')
xlim([0.06 0.16])
title(sprintf('PSD Percent Error | %s',variable));
legend

%% Moving Correlation Coefficient
nwin = 8000;

for i = 1:length(time)-nwin
    rsim(i) = getCorrelationCoefficient(ytest(i:i+nwin),ysim(i:i+nwin));
    rslx(i) = getCorrelationCoefficient(ytest(i:i+nwin),yslx(i:i+nwin));
    robs(i) = getCorrelationCoefficient(ytest(i:i+nwin),yobs(i:i+nwin));
end

figure
subplot(2,1,1)
gca; hold on; box on;
plot(time,ytest,'DisplayName','Experiment');
plot(time,ysim,'DisplayName','OpenFAST');
plot(time,yslx,'DisplayName','State-Space');

xlabel('Time [s]')
title({sprintf('Moving Correlation Coefficient | %s',variable)},{sprintf('Window Size: %0.3g seconds',nwin*0.0416)});
legend

subplot(2,1,2)
gca; hold on; box on;
plot(time(1:end-nwin),rslx,'DisplayName','State-Space');
plot(time(1:end-nwin),rsim,'DisplayName','OpenFAST');
plot(time(1:end-nwin),robs,'DisplayName','Kalman Filter')

xlabel('Time [s]')
legend

% % Mean Square Error
% slx_MSEi = slx_abs.^2;
% sim_MSEi = sim_abs.^2;
% 
% nwin = 10;
% winFreq = freq(1:end-nwin);
% for i = 1:length(test_psd(:,2))-nwin
%     slx_MSE(i) = sum(slx_MSEi(i:i+nwin,2));
%     sim_MSE(i) = sum(sim_MSEi(i:i+nwin,2));
% end
% 
% figure
% plot(winFreq,slx_MSE,'DisplayName','StateSpace');
% plot(winFreq,sim_MSE,'DisplayName','OpenFAST');
% 
% xlabel('Frequency [Hz]')
% xlim([0.04 0.18])
% title(sprintf('PSD Running Mean Square Error | %s | Window = %0.3g',variable,nwin/Fs));
% 
% % Root Mean Square Error
% slx_RMSE = slx_MSE.^0.5;
% sim_RMSE = sim_MSE.^0.5;
% 
% % Sum of squares fit
% nwin = 1;
% fbar = zeros(length(test_psd(:,2))-nwin,1);
% R = zeros(length(test_psd(:,2))-nwin,1);
% for i = 1:length(test_psd(:,2))-nwin
%     measured = test_psd(i:i+nwin,2);
%     predicted = sim_psd(i:i+nwin,2);
% 
%     R(i) = sumSquareFit(predicted,measured);
%     fbar(i) = mean(test_psd(i:i+nwin,1));
% end



% figure
% subplot(3,1,1)
% gca; hold on; box on;
% plot(sim_psd(:,1),sim_psd(:,2),'DisplayName','State-Space');
% plot(test_psd(:,1),test_psd(:,2),'DisplayName','Experiment');
% xlabel('Frequency [Hz]')
% xlim([0.05,0.15])
% title('Pitch PSDs')
% legend
% 
% subplot(3,1,2)
% gca; hold on; box on;
% plot(fbar,R);
% xlabel('Frequency [Hz]')
% ylim([0,1])
% xlim([0.05,0.15])
% title({'Moving R^2 Fit'},{sprintf('%i Samples',nwin)})
% 
% subplot(3,1,3)
% gca; hold on; box on;
% plot(sim_psd(:,1),relError);
% 
% xlabel('Frequency [Hz]')
% xlim([0.05 0.15])
% title('PSD Absolute Error')

%% Cross Correlation Analysis
[simr,simlags] = xcorr(ysim,ytest,'coeff');
[slxr,slxlags] = xcorr(yslx,ytest,'coeff');
[obsr,obslags] = xcorr(yobs,ytest,'coeff');

[simupper,simlower] = envelope(simr);
[slxupper,slxlower] = envelope(slxr);
[obsupper,obslower] = envelope(obsr);


figure
gca; box on; hold on;
plot(simlags,simupper,'DisplayName','OpenFAST');
plot(slxlags,slxupper,'DisplayName','StateSpace');
plot(obslags,obsupper,'DisplayName','Kalman Filter');

title({'Cross-Correlation Envelope'},{sprintf('%s vs. Experimental Results',variable)})
xlabel('Lags')
ylabel('Normalized Cross-Correlation')
legend
% 
% % %% Covariance
% % C = cov(ysim,ytest);
% % 
% % for i = 1:length(ysim)
% %     Cpoints(i) = cov(ysim(i),ytest(i));
% % end
% % 
% % figure
% % stem3(ysim,ytest,Cpoints)

%% Absolute Error Analysis
fast_absError = abs(ytest-ysim)/max(ytest);
ss_absError = abs(ytest-yslx)/max(ytest);
kalman_absError = abs(ytest-yobs)/max(ytest);

figure

gca; hold on; box on;
plot(time,fast_absError,'DisplayName','OpenFAST');
plot(time,ss_absError,'DisplayName','State-Space');
plot(time,kalman_absError,'DisplayName','Kalman Filter')

% subplot(2,1,2)
% gca; hold on; box on;
% plot(time,absMovError,'DisplayName','Moving Mean of Absolute Error');
% title('Relative Error Moving Mean')

% %% Cumulative Integral
% sim_trap = cumtrapz(time,ysim);
% test_trap = cumtrapz(time,ytest);
% 
% 
% 
% figure
% gca; box on; hold on;
% plot(time,absError,'DisplayName','Absolute Error');
% % plot(time,relError,'DisplayName','Relative Error');
% title('Pitch Error')
% legend
% 
% % %% Running Integral
% % nwin = 500;
% % 
% % sim_run = runInt(time,ysim,nwin);
% % test_run = runInt(time,ytest,nwin);
% % 
% % figure
% % gca; box on; hold on;
% % plot(rMean(sim_run),'DisplayName','Simulation')
% % plot(rMean(test_run),'DisplayName','Measured')
% % legend
% % title('Running Integral')

% %% Running Standard Deviation
% nwin = 5000;
% 
% sim_std = movmean(absError,nwin);
% test_std = movmean(ytest,nwin);
% 
% 
% test_std = smoothdata(test_std,'gaussian');
% 
% figure
% gca; box on; hold on;
% plot(time,sim_std,'DisplayName','Simulated');
% % plot(time,test_std,'DisplayName','Measured');
% legend
% title('Running Standard Deviation')








% ---------- FUNCTIONS ---------- %

function R = sumSquareFit(predicted,measured)
    ybar = mean(measured);
    fbar = mean(predicted);

    SSR = sum((measured-predicted).^2);
    SST = sum((measured-fbar).^2);
    R = 1 - SSR/SST;
end

function I = runInt(X,Y,window)
    for i = 1:length(X)-window
        start = i;
        stop = i+window;
        
        xwin = X(start:stop);
        ywin = Y(start:stop);

        I(i) = trapz(xwin,ywin);
    end
end

function I = windowInt(vals)
    I = trapz(vals)
end

function a = average(vals)

a = sum(vals)/length(vals);

end































% 
% dofs = {'PtfmPitch','TwrBsMyt'};
% index = 25000;
% 
% for i = 1:length(dofs)
%     slx_results.(dofs{i}) = pchip(slx_results.Time,slx_results.(dofs{i}),test_results.Time);
% end
% 
% %% Loop Through Fields
% for i = 1:length(dofs)
%     y1 = test_results.(dofs{i});
%     y2 = slx_results.(dofs{i});
%     time = test_results.Time;
%     time = time(1:25000);
% 
% 
%     % Residual
%     SSE = (y1(1:index)-y2(1:index)).^2;
%     SST = (y1(1:index)-mean(y1(1:index))).^2;
%     rsquare{i} = 1-SSE./SST;
% 
%     % Absolute Error
%     absE = abs(y1(1:index)-y2(1:index));
%     perE = absE./abs(y1(1:index));
% 
%     figure
%     title(dofs{i})
% %     plot(time,absE,'DisplayName','Error'); hold on
%     plot(time,abs(y1(1:index)),'DisplayName','Measured'); hold on;
%     plot(time-29.975,abs(y2(1:index)),'DisplayName','Simulated')
%     xlim([0 400])
%     legend
% end

end

