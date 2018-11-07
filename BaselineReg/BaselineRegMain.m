addpath(genpath('../'))
warning('off', 'all')

clear

%% Configuration

% import data
%importdata_Report1 % non-one-out-of-k-coded
if(version()==('9.5.0.944444 (R2018b)'))
    %data=load('../XoneoutofK.mat');
    data=load('../X.mat');
    X=data.X;
else
    %importdata_Report2; %For K out of N
    importdata_Report1; %For K
end

% p-norm:
p_dist = @(y,yM,p) 1/length(y) * sum(abs(y-yM).^p);

% Loss function:
% y is test data output, yM is what the model thinks it is
L = @(y,yM) p_dist(y, yM, 2);   % euclidian  p = 2

% Tree level analysis configuration
seed = 1; % random seed used for crossval splits

% cross validation configuration
Kouter = 5;
Kinner = 5;

% generate splits
[outer_train_cell, inner_train_cell] = genSplits(X, Kouter, Kinner, seed);

% which argument is output?
outarg = 1; % id of the X(:,id) data matrix. 1: gpm

% features available are all the ones that are not the output attribute
% this is a vector of indices in X(:,index):
features_avail = 2:8; % no year as it is not useful
features = features_avail( ~ismember(features_avail, outarg) );

% model functions
Train = @(     X) BaselineRegTrain(X, outarg); % 1 stands for first order reg
Exe   = @(par, X) BaselineRegExecute(par, X);

% compute
[Egen] = crossvalidate(X, {Train}, {Exe}, L, outarg, outer_train_cell, inner_train_cell);

par = Train(X);

%% output

disp(' ')
disp('|----- Calculations finished -----|')
disp(' ')
disp(strcat('Estimated generalisation error: ', num2str(Egen)))
disp(' ')
