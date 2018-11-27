%%KNN Density
x = linspace(-2, 2, 50)';
%mfig('Histogram'); clf;
hist(X);
%%
% Number of neighbors
K = 10;

% x-values to evaluate the GMM
x = ones(100,6);
x = reshape(-3:3,100,6)

% Find the k nearest neighbors
[i,D] = knnsearch(X, x, 'K', K);

% Compute the density
density = 1./(sum(D,2)/K);

% Plot KNN estimate of density
%mfig('KNN density'); clf;
plot(x, density);
%%
% Compute the average relative density
[iX,DX] = knnsearch(X, X, 'K', K+1);
densityX= 1./(sum(DX(:,2:end),2)/K);
avg_rel_density=density./(sum(densityX(i(:,2:end)),2)/K);


% Plot KNN estimate of density
mfig('KNN average relative density'); clf;
plot(x, avg_rel_density);