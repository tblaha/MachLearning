%% Load data
addpath(genpath('../'))
warning('off', 'all')

clear
%%
out = fitgmdist(X,4,'Options')
%% Configuration

% import data
%importdata_Report1 % non-one-out-of-k-coded
if(version()==('9.5.0.944444 (R2018b)'))
    %data=load('../XoneoutofK.mat');
    data=load('X.mat');
    X=data.X;
else
    %importdata_Report2; %For K out of N
    importdata_Report1; %For K
end
%L = @(y,yM) bayesloss(y,yM);
%% No feature selection, just use all.
% But do some numerical analysis to figure out a suitable tree size

% which argument is output?
outarg = 8; % id of the X(:,id) data matrix. 1: gpm

% features available are all the ones that are not the output attribute
% this is a vector of indices in X(:,index):
features_avail = 1:6; % no year as it is not useful
features = features_avail( ~ismember(features_avail, outarg) );

%%
    % selecting outputs
    classNames= {'USA', 'Europe', 'Asia'}';
    y=zeros(size(X,1),1);

    if(size(X,2)>8)     %If X one out of K is runned, else it's normal X
        for i = 1:size(X,1)
            if(X(i,8)==1)
                y(i)=1;
            elseif(X(i,9)==1)
                y(i)=2;
            elseif(X(i,10)==1)
                y(i)=3;
            end
        end
        for i=1:4   %Removing origins and year
            X(:,7)=[];
        end
    else
       y=X(:,outarg);    %Saving the origins to be used for classNames
    end
    
    % select inputs
    attributeNames = {'gpm', 'cylinders', 'displacement', 'horsepower', 'weight', 'accelleration'};
    features = features( ~ismember(features, outarg) ); % extra safety...
    X = X(:,features);

%%
[N,M]=size(X);

%%
% exercise 11.3.2

% Kernel width
%w = 5;

% Estimate optimal kernel density width by leave-one-out cross-validation
widths=max(var(X))*(2.^[-10:2]);
for w=1:length(widths)
    [f, log_f] = gausKernelDensity(X, widths(w));
    logP(w)=sum(log_f);
end
[val,ind]=max(logP);
width=widths(ind);
disp(['Optimal estimated width is ' num2str(width)])

% Estimate density for each observation not including the observation
% itself in the density estimate
f = gausKernelDensity(X, width);

% Sort the densities
[y,i] = sort(f);

% Display the index of the lowest density data object
% The outlier should have index 1001
disp(i(1));

% Plot density estimate outlier scores
%figure('Gaussian Kernel Density: outlier score'); clf;
bar(y(1:20));

%mfig('Gaussian Kernel Density: Possible outliers'); clf;
%%
Px = [...
-0.3987    0.0631    0.2064   -0.1998    0.4816   -0.6655    0.2280   -0.1664
   -0.4012   -0.1338   -0.0519   -0.1853   -0.6865   -0.2436    0.2696    0.4251
   -0.4150   -0.1222   -0.0550   -0.1084   -0.3053    0.2082   -0.0755   -0.8098
   -0.4018    0.1217   -0.2130   -0.1081    0.3561    0.5986    0.4955    0.1899
   -0.3999   -0.2091    0.0124   -0.2981    0.2020    0.1276   -0.7477    0.3019
    0.2633   -0.4353    0.6430   -0.4780    0.0076    0.2274    0.2176   -0.0136
    0.2092   -0.6684   -0.6418   -0.1573    0.1683   -0.1625    0.1213   -0.0582
    0.2710    0.5181   -0.2842   -0.7485   -0.0865   -0.0406   -0.0548   -0.0702];

PCAs = X*Px(:,[1,2]);
%%
%% Reference plots

figCyl = figure('Position', [100 100 800 500], 'Visible', 'off');
hold on
for k = 1:2
    scatter(scat(ismember(cyls,cylgroups{k}),1),...
            scat(ismember(cyls,cylgroups{k}),2),... 'b',...
            'DisplayName', strcat('Cylinder: ', mat2st  r(cylgroups{k}')))
end
hold off
title('Scatter Plot for the raw data -- Number of Cylinders', 'FontSize', 16)
xlabel('PCA1 score', 'FontSize', 16)
ylabel('PCA2 score', 'FontSize', 16)
grid on
axis equal
hold off
legend({}, 'Location', 'SouthWest', 'Fontsize', 14)
saveas(figCyl, 'Plots/ScatCyl.eps', 'epsc')

%%
scat=PCAs;
figOri = figure('Position', [100 100 800 500], 'Visible', 'off');
hold on
for k = 1:2
    scatter(scat(X == k, 1),...
            scat(X == k, 2),... 'b',...
            'DisplayName', strcat('Origin: ', k));
end
hold off
title('Scatter Plot for the raw data -- Origin', 'FontSize', 16)
xlabel('PCA1 score', 'FontSize', 16)
ylabel('PCA2 score', 'FontSize', 16)
grid on
axis equal
hold off
legend({}, 'Location', 'SouthWest', 'Fontsize', 14)
%%saveas(figOri, 'Plots/ScatOri.eps', 'epsc')

%%
for k = 1:20
    subplot(4,5,k);
    imagesc(reshape(X(i(k),:), 16, 16)); 
    title(k);
    colormap(1-gray); 
    axis image off;
end
