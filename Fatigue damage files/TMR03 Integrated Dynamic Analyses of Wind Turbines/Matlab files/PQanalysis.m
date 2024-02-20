function [P,Q,fighandle,xbar,dx,tp] = PQanalysis(t,x,plotflag)

% PQ analysis
% Inputs: time series t and motion history x. Note that x should have zero
% mean! 
% Outputs: coefficients P and Q, figure handle, raw data xbar and dx, and 
% tp (peaks and troughs identified by the code, first column time, second
% column is x values)


if plotflag
    fighandle = figure(); 
    set(gcf,'Position',[200 200 600 300])
    subplot(1,2,1)
    plot(t,x)
    xlabel('$t$','Interpreter','Latex')
    ylabel('$x$','Interpreter','Latex')
    grid on
    
end

% find the turning points using WAFO
tp = dat2tp([t,x]);
tp = tp(2:end,:);
tpp = tp(tp(:,2)>0,:);
hold on
plot(tpp(:,1),tpp(:,2),'k.')

% find mean values and differences for positive peaks
xbar = 0.5*( tpp(2:end,2)+tpp(1:end-1,2));
dx = ( tpp(1:end-1,2)-tpp(2:end,2))./xbar;

% find polynomial fit 
pcoeffs = polyfit(xbar,dx,1);
P = pcoeffs(2);
Q = pcoeffs(1);

if plotflag
    subplot(1,2,2)
    plot(xbar,dx,'.')
    xlabel('$\bar{x}$','Interpreter','Latex')
    ylabel('$\frac{x_i-x_{i+1}}{\bar{x}}$','Interpreter','Latex')
    grid on
    hold on
    plot(xbar,polyval(pcoeffs,xbar))
    legend('measured','fitted')
end