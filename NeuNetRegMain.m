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
features = [2 3 4 5 6 7 8];

% don't do feature selection, just take all. Instead, do complexity control
hmax = 14;
hes = 10:hmax;
%hes = 0;
Egen_list = zeros(length(hes),1);

tic
for i = 1:length(hes) % least complex to most complex
    hiddenlayers = hl_try(hes(i));
    Train = @(     X) NeuNetRegTrain  (X,   hiddenlayers, features, outarg); % 1 stands for first order reg
    Exe   = @(par, X) NeuNetRegExecute(par, X  , features, outarg);
       [Egen_list(i)] = crossvalidate(X, {Train}, {Exe}, L, outarg, Kouter, Kinner, seed);
    disp(strcat( "Top layer just finished" , num2str(hes(i)/hmax *100), "%"))
end
toc

[~,idx] = min(Egen_list);
Egen = Egen_list(idx);

par_best = NeuNetRegTrain(X, hl_try(hes(idx)), features, outarg);



%% output


disp(' ')
disp('|----- Calculations finished -----|')
disp(' ')
disp(strcat('Selected number of hidden neurons: ', mat2str(hl_try(hes(idx)))))
disp(' ')
disp(strcat('Estimated generalisation error: ', num2str(Egen)))
disp(' ')

function hiddenlayers = hl_try(hlnum)

    if hlnum == 0
        hiddenlayers = [];
    elseif hlnum <= 3
        hiddenlayers = [hlnum];
    elseif hlnum > 3 && hlnum <= 8
        lastl = min(round(hlnum/2),3);
        hiddenlayers = [hlnum - lastl,lastl];
    elseif hlnum > 8
        lastl = 3;
        hlleft = hlnum - lastl;
        hlmid = min(round(hlleft/2), 5);
        hiddenlayers = [hlleft - hlmid, hlmid, lastl];
    end
            
        
end

