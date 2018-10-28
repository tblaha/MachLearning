addpath(genpath('./'))
warning('off', 'all')

clear

%% Configuration

% import data
importdata_Report1 % non-one-out-of-k-coded

L = @(y,yM) bayesloss(y,yM);


% model functions
Train = @(     X, feats, o) NaiveBayesTrain  (     X, feats, o); % 1 stands for first order reg
Exe   = @(par, X, feats, o) NaiveBayesExecute(par, X, feats, o);


% cross validation configuration
Kouter = 5;
Kinner = 5;


% fwd features selection configuration
seed = 10; % random seed used for crossval splits
errortolerance = 0.0001; % see function documentation of FwdFeatSel



%% Fwd features selection

% which argument is output?
outarg = 8; % id of the X(:,id) data matrix. 1: gpm

% features available are all the ones that are not the output attribute
% this is a vector of indices in X(:,index):
features_avail = 1:length(X(1,:));
features_avail = features_avail( ~ismember(features_avail, outarg) );

% run the fwd feature selection:
tic %measure time
    % function [features, stoppingCriteria] = FwdFeatSel(features_avail, X, TrainFcn, ExeFcn, LossFcn, outarg, ErrorTol, Kouter, Kinner)
    [features, StoppingCriteria] = FeatSel(features_avail, 'bwd', X, Train, Exe, L, outarg, errortolerance, Kouter, Kinner, seed);
toc %measure time



%% Print stopping reason
disp(' ')
disp(StoppingCriteria)
disp(' ')



%% Evaluate the best best model

% for testing purposes

P = cell(1);
M = cell(1);
    
P{1} = @(X)      Train  (X, features, outarg);
M{1} = @(par, X) Exe    (par, X, features, outarg);

% "crossvalidate" with only one model --> just to get generalization error
[Egen, ~] = crossvalidate(X, P, M, L, outarg, Kouter, Kinner, seed);



%% output

disp(' ')
disp('|----- Calculations finished -----|')
disp(' ')
disp(strcat('Selected model features: ', mat2str(features)))
disp(' ')
disp(strcat('Estimated generalisation error: ', num2str(Egen)))
disp(' ')

%% logistic loss function

function loss = bayesloss(y,yM) % y is 8th and 9th column
    
    %ySingle = (y(:,1) == 1) *1 + (y(:,2) == 1) *2 + (y(:,1) == 0 & y(:,2) == 0) *3; % code back from one out of k
    loss = sum(abs((y - yM)) > 0) / length(y);

end
