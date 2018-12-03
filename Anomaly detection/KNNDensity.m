%import data

%if(version()==('9.5.0.944444 (R2018b)'))
    %data=load('XoneoutofK.mat');
    data=load('X.mat');
    X=data.X;
    %datann=load('Xnn.mat'); % not normalizes values
    %Xnn=datann.Xnn;
%else
    %run(importdata_Report2.m); %For K out of N
%    run(importdata_Report1.m); %For K
%end

% Number of neighbors
K = 6;


% y-values to evaluate the GMM

%Y = X(300:end,:);

% Find the k nearest neighbors
[i,D] = knnsearch(X, X, 'K', K+1);

% Compute the density
density = 1./(sum(D,2)/K);

[y,id]=sort(density);

mfig('Y'); clf;
bar(y(1:20));
title('KNN Density: outlier score')
xlabel('Index')
set(gca,'XTick',1:20,'XTickLabel',id,'FontSize',14)
% Compute the average relative density

[iX,DX] = knnsearch(X, X, 'K', K+1);
densityX= 1./(sum(DX(:,2:end),2)/K);
avg_rel_density=densityX./(sum(densityX(i(:,2:end)),2)/K);

[y_avg,id_avg]=sort(avg_rel_density);

mfig('Y_avg'); clf;
bar(y_avg(1:20));
title('KNN ARD Density: outlier score')
xlabel('Index')
set(gca,'XTick',1:20,'XTickLabel',id_avg,'FontSize',14)
% Plot KNN estimate of density

mfig('KNN density'); clf;
plot(density(1:100));

% Plot KNN estimate of density
mfig('KNN average relative density'); clf;
plot(avg_rel_density(1:100));