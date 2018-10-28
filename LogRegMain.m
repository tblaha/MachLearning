addpath(genpath('./'))
warning('off', 'all')

clear

%% Configuration

% import data
importdata_Report2 % one-out-of-k-coded

X = X(:,1:9); % reduced one-out-of-k-coded (ommiting the last one for computational performance)

%L = @(y,yM) ( sum(yM - y > 0) * r + sum(yM - y < 0) * (1-r) ) / length(y);
L = @(y,yM) logiloss(y,yM);



% model functions
Train = @(X, feats, o)      LogRegTrain  (    X, feats, o); % 1 stands for first order reg
Exe   = @(par, X, feats, o) LogRegExecute(par,X, feats, o);


% cross validation configuration
Kouter = 5;
Kinner = 5;


% fwd features selection configuration
seed = 12; % random seed used for crossval splits
errortolerance = 0.0001; % see function documentation of FwdFeatSel



%% Fwd features selection

% which argument is output?
outarg = 8:9; % id of the X(:,id) data matrix. 1: gpm

% features available are all the ones that are not the output attribute
% this is a vector of indices in X(:,index):
features_avail = 1:length(X(1,:));
features_avail = features_avail( ~ismember(features_avail, outarg) );

% run the fwd feature selection:
tic %measure time
    % function [features, stoppingCriteria] = FwdFeatSel(features_avail, X, TrainFcn, ExeFcn, LossFcn, outarg, ErrorTol, Kouter, Kinner)
    [features, StoppingCriteria] = FeatSel(features_avail, 'fwd', X, Train, Exe, L, outarg, errortolerance, Kouter, Kinner, seed);
toc %measure time



%% Print stopping reason
disp(' ')
disp(StoppingCriteria)
disp(' ')



%% Evaluate the best best model

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

function loss = logiloss(y,yM) % y is 8th and 9th column
    
    ySingle = (y(:,1) == 1) *1 + (y(:,2) == 1) *2 + (y(:,1) == 0 & y(:,2) == 0) *3; % code back from one out of k
    
    
    loss = sum(abs((ySingle - yM)) > 0) / length(y);

end
