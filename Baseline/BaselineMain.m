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
    %mportdata_Report2; %For K out of N
    importdata_Report1; %For K
end

% loss fcn
L = @(y,yM) bayesloss(y,yM); 


% cross validation configuration
Kouter = 5;
Kinner = 5;


% Tree level analysis configuration
seed = 2; % random seed used for crossval splits

% generate splits
[outer_train_cell, inner_train_cell] = genSplits(X, Kouter, Kinner, seed);


%% No feature selection, just use all.
% But do some numerical analysis to figure out a suitable tree size

% which argument is output?
outarg = 8; % id of the X(:,id) data matrix. 1: gpm

% features available are all the ones that are not the output attribute
% this is a vector of indices in X(:,index):
features_avail = 1:6; % no year as it is not useful
features = features_avail( ~ismember(features_avail, outarg) );


% model functions
Train = @(     X) BaselineTrain(X, outarg); % 1 stands for first order reg
Exe   = @(par, X) BaselineExecute(par, X);

% compute
[Egen,~,Etest] = crossvalidate(X, {Train}, {Exe}, L, outarg, outer_train_cell, inner_train_cell);

%% output

disp(' ')
disp('|----- Calculations finished -----|')
disp(' ')
disp(strcat('Estimated generalisation error: ', num2str(Egen)))
disp(' ')

%% logistic loss function

function loss = bayesloss(y,yM) % y is 8th and 9th column
    
    %ySingle = (y(:,1) == 1) *1 + (y(:,2) == 1) *2 + (y(:,1) == 0 & y(:,2) == 0) *3; % code back from one out of k
    loss = sum(abs((y - yM)) > 0) / length(y);

end
