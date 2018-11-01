addpath(genpath('./'))
warning('off', 'all')

clear

%% Configuration

% import data
importdata_Report1 % non-one-out-of-k-coded

% p-norm:
p_dist = @(y,yM,p) 1/length(y) * sum(abs(y-yM).^p);

% Loss function:
% y is test data output, yM is what the model thinks it is
L = @(y,yM) p_dist(y, yM, 2);   % euclidian  p = 2
%L = @(y,yM) p_dist(y, yM, 1);   % city block p = 1

% model functions will be configured later

% cross validation configuration
Kouter = 5;
Kinner = 5;

% complexity control parameters
seed = 4; % random seed used for crossval splits
errortolerance = 0.001; % see function documentation of FeatSel, works well



%% Fwd features selection

% which argument is output?
outarg = 1; % id of the X(:,id) data matrix. 1: gpm

% features available are all the ones that are not the output attribute
% this is a vector of indices in X(:,index):
features_avail = 1:length(X(1,:));
features = features_avail( features_avail ~= outarg );
features = [4 5 7];

% don't do feature selection, just take all. Instead, do complexity control
hiddenlayers = [5 3];
Train = @(     X) NeuNetRegTrain  (X,   hiddenlayers, features, outarg); % 1 stands for first order reg
Exe   = @(par, X) NeuNetRegExecute(par, X  , features, outarg);
tic
    [Egen] = crossvalidate(X, {Train}, {Exe}, L, outarg, Kouter, Kinner, seed);
toc

par_best = Train(X);


%% output

disp(' ')
disp('|----- Calculations finished -----|')
disp(' ')
disp(strcat('Selected model features: ', mat2str(features)))
disp(' ')
disp(strcat('Estimated generalisation error: ', num2str(Egen)))
disp(' ')

