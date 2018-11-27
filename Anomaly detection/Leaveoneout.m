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
% exercise 11.3.2

% Kernel width
w = 5;

% Estimate optimal kernel density width by leave-one-out cross-validation
widths=2.^[-10:10];
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
%figure('Outlier score'); clf;
bar(y(1:20));
