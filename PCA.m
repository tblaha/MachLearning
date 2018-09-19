clear
close all

%%
importdata

%% linear algebra
[~,D_raw] = eig(X'*X); % raw eigenvalues from the var-covar matrix
sigmas = sort(diag(D_raw), 'desc'); % sorted eigenvalues

corr  = corr(X);   % variance normalized variance covariance matrix (correlation matrix)
[P,D] = eig(corr); % eigenvectors (P) and eigenvalues diag(D) or the corr mat

%% plots
% plot variance explaination plot
plot( 0:M, [0; cumsum(diag(D) / trace(corr)) ], '-o')
title('Variance Explaned')
xlabel('Number of Principal Components')
ylabel('Cumulative Normalized Variance')
grid on
ylim([0,1])