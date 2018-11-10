addpath(genpath('../'))
warning('off', 'all')

clear

%% Configuration

% import data
if(version()==('9.5.0.944444 (R2018b)'))
    %data=load('../XoneoutofK.mat');
    data=load('../X.mat');
    X=data.X;
else
    %importdata_Report2; %For K out of N
    importdata_Report1; %For K
end

L = @(y,yM) bayesloss(y,yM);


% model functions
Train = @(     X, feats, o) NaiveBayesTrain  (     X, feats, o); % 1 stands for first order reg
Exe   = @(par, X, feats, o) NaiveBayesExecute(par, X, feats, o);


% fwd features selection configuration
seed = 4; % random seed used for crossval splits
errortolerance = 0.0001; % see function documentation of FwdFeatSel

% cross validation configuration
Kouter = 5;
Kinner = 5;

% generate splits
[outer_train_cell, inner_train_cell] = genSplits(X, Kouter, Kinner, seed);


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
    [features, StoppingCriteria, Egen_list, Etests] = FeatSel(features_avail, 'bwd', X, Train, Exe, L, outarg, errortolerance, outer_train_cell, inner_train_cell);
toc %measure time
%features=[2 3 4];
%Train = @(     X) NaiveBayesTrain  (     X, features, outarg); % 1 stands for first order reg
%Exe   = @(par, X) NaiveBayesExecute(par, X, features, outarg);
    
    % check best model
    %[Egen_ist] = crossvalidate(X, {Train}, {Exe}, L, outarg, outer_train_cell, inner_train_cell);


figure('Name', 'Generalization Error')
plot(Egen_list)


figure('Name', 'Test Errors')
plot(Etests)


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
[Egen, ~] = crossvalidate(X, P, M, L, outarg, outer_train_cell, inner_train_cell);



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
