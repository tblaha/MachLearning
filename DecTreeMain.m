addpath(genpath('./'))
warning('off', 'all')

clear

%% Configuration

% import data
%importdata_Report1 % non-one-out-of-k-coded
if(version()==('9.5.0.944444 (R2018b)'))
    %data=load('XoneoutofK.mat');
    data=load('X.mat');
    X=data.X;
else
    %run(importdata_Report2.m); %For K out of N
    run(importdata_Report1.m); %For K
end
L = @(y,yM) bayesloss(y,yM);


% model functions
% --> will be declared in for loop



% cross validation configuration
Kouter = 5;
Kinner = 5;


% Tree level analysis configuration
seed = 2; % random seed used for crossval splits
errortolerance = 0.0001;



%% No feature selection, just use all.
% But do some numerical analysis to figure out a suitable tree size

% which argument is output?
outarg = 8; % id of the X(:,id) data matrix. 1: gpm

% features available are all the ones that are not the output attribute
% this is a vector of indices in X(:,index):
features_avail = 1:6; % no year as it is not useful
features = features_avail( ~ismember(features_avail, outarg) );

Egen_hist = [];
minpar_hist = [70]; % the lower, the more complex the tree
alpha = 0.8; % decrease minpar by (roughly) (1-alpha) *100 percent every iteration
i = 1;
strikes = 10; % how many consecutive worse iterations to tolerate

while true    
    % model functions
    Train = @(     X) DecTreeTrain  (     X, features, outarg, minpar_hist(i)); % 1 stands for first order reg
    Exe   = @(par, X) DecTreeExecute(par, X, features, outarg);
    
    % check best model
    [Egen_hist(i)] = crossvalidate(X, {Train}, {Exe}, L, outarg, Kouter, Kinner, seed);
     
    % new minpar
    minpar_hist(i+1) = round(minpar_hist(i)*alpha);
    
    % status update
    disp(i)
    disp(Egen_hist(i))
    
    % stopping criterium
    if i > strikes && sum(Egen_hist(i-strikes) - Egen_hist(i-strikes+1:i) < errortolerance * ones(1,strikes)) == strikes
        % this means that for strikes iterations already no improvements 
        minpar = minpar_hist(i-strikes);
        Egen   = Egen_hist(i-strikes);
        break;
    end
    
    % counting
    i = i + 1;
       
end

figure('Name', 'Generalization Error')
plot(minpar_hist(1:end-1),Egen_hist)
grid on
set(gca, 'XDir','reverse')

% return tree
outtree = DecTreeTrain  (X, features, outarg, minpar);

%% output

disp(' ')
disp('|----- Calculations finished -----|')
disp(' ')
disp(strcat('Best minimum parents level: ', mat2str(minpar)))
disp(' ')
disp(strcat('Estimated generalisation error: ', num2str(Egen)))
disp(' ')

%% logistic loss function

function loss = bayesloss(y,yM) % y is 8th and 9th column
    
    %ySingle = (y(:,1) == 1) *1 + (y(:,2) == 1) *2 + (y(:,1) == 0 & y(:,2) == 0) *3; % code back from one out of k
    loss = sum(abs((y - yM)) > 0) / length(y);

end
