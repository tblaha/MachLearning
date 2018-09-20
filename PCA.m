clear
close all



%%
importdata



%% linear algebra
[~,D_raw] = eig((X-mean(X))'*(X-mean(X))); % raw eigenvalues from the var-covar matrix
sigmas = sort(diag(D_raw), 'desc'); % sorted eigenvalues

corrmat  = corr(X);   % variance normalized variance covariance matrix (correlation matrix)
[P,D]    = eig(corrmat); % eigenvectors (P) and eigenvalues diag(D) or the corr mat

corrmaty = corr(Y);
[Py,Dy]  = eig(corrmaty);



%% plots
% plot variance explaination plot
varplot = figure('Position',[0 0 800 600],'visible','off');
hold on
plot( 0:M, [0; cumsum(diag(D) / trace(corrmat)) ], '-o')
plot( 0:M-1, [0; cumsum(diag(Dy) / trace(corrmaty)) ], '-x')
hold off
title('Variance Explaned')
xlabel('Number of Principal Components')
ylabel('Cumulative Normalized Variance')
grid on
ylim([0,1])
legend('Cumulative \sigma of X', 'Cumulative \sigma of Y')
saveas(varplot, strcat('Plots/varplot.eps'),'epsc')

% group by continent:
X1 = X(X(:, end) == 1, :);
X2 = X(X(:, end) == 2, :);
X3 = X(X(:, end) == 3, :);
Y1 = Y(Y(:, end) == 1, :);
Y2 = Y(Y(:, end) == 2, :);
Y3 = Y(Y(:, end) == 3, :);

% plot 1st and 2nd principal components of corr(X)
prinplot12 = figure('Position',[0 0 800 600],'visible','off');
hold on
plot( (X1-mean(X1)) * P(:,1), (X1-mean(X1)) * P(:,2), 'o')
plot( (X2-mean(X2)) * P(:,1), (X2-mean(X2)) * P(:,2), 'x')
plot( (X3-mean(X3)) * P(:,1), (X3-mean(X3)) * P(:,2), '*')
hold off
grid on
title('Princial Component Analysis -- First 2 components')
xlabel('X^T v_1 -- 1st principal component projections of X')
ylabel('X^T v_2 -- 2st principal component projections of X')
legend('American', 'European', 'Asian')
saveas(prinplot12, strcat('Plots/1stPrin-2ndPrin.eps'),'epsc')

% plot 1st principal components of corr(Y) and output
prinplot1out  = figure('Position',[0 0 800 600],'visible','off');
hold on
plot( (Y1-mean(Y1)) * Py(:,1), X1(:,1), 'o')
plot( (Y2-mean(Y2)) * Py(:,1), X2(:,1), 'x')
plot( (Y3-mean(Y3)) * Py(:,1), X3(:,1), '*')
hold off
grid on
title('Princial Component Analysis -- Output vs first component')
xlabel('Y^T v_1 -- 1st principal component projections of Y')
ylabel('Output [gallons per mile]')
legend('American', 'European', 'Asian')
saveas(prinplot1out, strcat('Plots/1stPrin-Output.eps'),'epsc')

% plot 1st and 2nd principal component of corr(Y) and output. Save this one
% manually after selecting some nice viewing directions!
prinplot12out = figure('Position',[0 0 800 600],'visible','off');
hold on
plot3( (Y1-mean(Y1)) * Py(:,1), (Y1-mean(Y1)) * Py(:,2), X1(:,1), 'o')
plot3( (Y2-mean(Y2)) * Py(:,1), (Y2-mean(Y2)) * Py(:,2), X2(:,1), 'x')
plot3( (Y3-mean(Y3)) * Py(:,1), (Y3-mean(Y3)) * Py(:,2), X3(:,1), '*')
hold off
grid on
title('Princial Component Analysis -- Output vs first 2 components')
xlabel('Y^T v_1 -- 1st principal component projections of Y')
ylabel('Y^T v_2 -- 2st principal component projections of Y')
zlabel('Output [gallons per mile]')
legend('American', 'European', 'Asian')




