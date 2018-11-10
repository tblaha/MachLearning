clear
close all
%% this tests for significant model improvments for the regressions

Etest{1} = [0.3846    0.3418    0.3924    0.3590    0.3974]; % Baseline
Etest{2} = [0.2436    0.2532    0.2532    0.2051    0.1795]; % KNN
Etest{3} = [0.1538    0.1392    0.1392    0.1923    0.1667]; % DecTree
z(1,:) = Etest{1} - Etest{2}; % KNN vs Baseline
z(2,:) = Etest{2} - Etest{3}; % DecTree vs KNN
z(3,:) = Etest{1} - Etest{3}; % DecTree vs Baseline

name{1} = {'KNN', 'Baseline'};
name{2} = {'DecTree', 'KNN'};
name{3} = {'DecTree', 'Baseline'};

alpha = 0.1;



for i = 1:size(z,1)
    
    %% calculate parameters
    K = size(z,2);
    v = K - 1;

    zbar = mean(z(i,:));
    sigbar = sqrt( sum( (z(i,:) - zbar).^2 ./ (K*(K-1)) )  );

    %% do significance tests
    
    zL = fzero(@(x) integral( @(tau) nonstand_tpdf(tau, v, zbar, sigbar), -inf, x) - alpha/2      , 0);
    zU = fzero(@(x) integral( @(tau) nonstand_tpdf(tau, v, zbar, sigbar), -inf, x) - (1 - alpha/2), 0);
    ran = zU - zL;

    % probability that mean is in the left tail of the pdf, below 0 (aka the
    % LinReg model is actually better)
    pval_meanlargerzero = integral( @(tau) nonstand_tpdf(tau, v, zbar, sigbar), -inf, 0) ./ integral( @(tau) nonstand_tpdf(tau, v, zbar, sigbar), -inf, inf);


    %% Generate plots
    
    % sample the pdf
    n = 1001;
    fac = 0.3;
    x      = linspace( min(zL - fac*ran, -0.05), zU + fac*ran,  n);
    ps     = nonstand_tpdf(x, v, zbar, sigbar);

    sigplot = figure('Position', [100 100 1000 500], 'Visible', 'off');
    grid on
    hold on
        plot(x, ps, 'LineWidth', 2)
        plot([zL, zL], [0, 1.0*max(ps)], 'r--', 'LineWidth', 2)
        plot([zU, zU], [0, 1.0*max(ps)], 'r--', 'LineWidth', 2)
        % p value area
        xarea = linspace(min(zL - fac*ran, -0.1), 0, 101);
        psarea = nonstand_tpdf(xarea, v, zbar, sigbar);
        h = area( xarea, psarea, 'LineStyle', 'none'); % for fancyness
        h(1).FaceColor = [0 0.447 0.741];
        % p value annotation
        an_x = (-x(1)+0.15) / (-x(1)+x(end)+0.3);
        annotation( 'textarrow', [0.375 an_x], [0.3 0.1175], 'String', strcat({'p-value = '}, num2str(pval_meanlargerzero, 3)), 'FontSize', 16)
    hold off
    xlim([-0.05, zU + fac*ran])
    legend({'p( E^g_A-E^g_B | (E^t_A-E^t_B)_k )', strcat('z_L and z_U (\alpha = ', num2str(alpha) ,')')}, 'Location', 'NorthWest', 'FontSize', 16)
    title(strcat({'Significance that '}, name{i}{1},{' is better than '},  name{i}{2}), 'FontSize', 18)
    xlabel('z', 'FontSize', 14)
    ylabel('non-standardized probability', 'FontSize', 14)

    saveas(sigplot, strcat('Plots/Sig_', name{i}{1}, '_vs_', name{i}{2}, '.eps'), 'epsc')
end
%% function

function p = nonstand_tpdf(x,v,mu,sig)
    const = gamma((v+1)/2) / (gamma(v/2) * sqrt(pi*v*sig^2) );
    p = const * (1 + 1/v * ((x-mu)/sig).^2).^(-(v+1)/2);
end
