clear
close all
%% this tests for significant model improvments for the regressions

Etest{1} = [0.9973    1.0468    1.0978    0.9206    0.9236]; % Baseline
Etest{2} = [0.1399 0.1079 0.1279 0.1022 0.1567]; % lin reg
Etest{3} = [0.1068    0.0873    0.0978    0.1054    0.1135]; % ANN
z(1,:) = Etest{1} - Etest{2}; % LinReg vs Baseline
z(2,:) = Etest{2} - Etest{3}; % ANN vs LinReg
z(3,:) = Etest{1} - Etest{3}; % ANN vs Baseline

name{1} = {'LinReg', 'Baseline'};
name{2} = {'ANN', 'LinReg'};
name{3} = {'ANN', 'Baseline'};

alpha = 0.05;



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
    x      = linspace( min(zL - fac*ran, -0.1), zU + fac*ran,  n);
    ps     = nonstand_tpdf(x, v, zbar, sigbar);

    sigplot = figure('Position', [100 100 1000 500], 'Visible', 'off');
    grid on
    hold on
        plot(x, ps, 'LineWidth', 1.5)
        plot([zL, zL], [0, 1.0*max(ps)], 'r--', 'LineWidth', 1.5)
        plot([zU, zU], [0, 1.0*max(ps)], 'r--', 'LineWidth', 1.5)
            xarea = linspace(min(zL - fac*ran, -0.1), 0, 101);
            psarea = nonstand_tpdf(xarea, v, zbar, sigbar);
            h = area( xarea, psarea, 'LineStyle', ':'); % for fancyness
            h(1).FaceColor = [0 0.447 0.741];
        an_x = (-x(1)+0.2) / (-x(1)+x(end)+0.35);
        annotation( 'textarrow', [0.375 an_x], [0.3 0.1175], 'String', strcat('p-value = ', num2str(pval_meanlargerzero)), 'FontSize', 16)
    hold off
    legend({'p( E^g_A-E^g_B | (E^t_A-E^t_B)_k )', strcat('z_L and z_U (\alpha = ', num2str(alpha) ,')')}, 'Location', 'NorthWest', 'FontSize', 12)
    title(strcat('Significance that ', name{i}{1},' is better than ',  name{i}{2}, ': pval = ', num2str(pval_meanlargerzero)), 'FontSize', 14)
    xlabel('z', 'FontSize', 14)
    ylabel('non-standardized probability', 'FontSize', 14)

    saveas(sigplot, strcat('Plots/Sig_', name{i}{1}, '_vs_', name{i}{2}, '.eps'), 'epsc')
end
%% function

function p = nonstand_tpdf(x,v,mu,sig)
    const = gamma((v+1)/2) / (gamma(v/2) * sqrt(pi*v*sig^2) );
    p = const * (1 + 1/v * ((x-mu)/sig).^2).^(-(v+1)/2);
end
