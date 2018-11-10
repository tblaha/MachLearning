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

% p-norm:
p_dist = @(y,yM,p) 1/length(y) * sum(abs(y-yM).^p);

% Loss function:
% y is test data output, yM is what the model thinks it is
L = @(y,yM) p_dist(y, yM, 2);   % euclidian  p = 2
%L = @(y,yM) p_dist(y, yM, 1);   % city block p = 1

% model functions
Train = @(X, feats, o)     LinRegTrain  (X, 1, feats, o); % 1 stands for first order reg
Exe   = @(par,X, feats, o) LinRegExecute(par,X,feats,o);


% fwd features selection configuration
seed = 1; % random seed used for crossval splits
errortolerance = 0.001; % see function documentation of FeatSel, works well

% cross validation configuration
Kouter = 5;
Kinner = 5;

% generate splits
[outer_train_cell, inner_train_cell] = genSplits(X, Kouter, Kinner, seed);




%% Fwd features selection

% which argument is output?
outarg = 1; % id of the X(:,id) data matrix. 1: gpm

% features available are all the ones that are not the output attribute
% this is a vector of indices in X(:,index):
features_avail = 1:length(X(1,:));
features_avail = features_avail( features_avail ~= outarg );

% run the fwd feature selection:
tic %measure time
    % function [features, stoppingCriteria] = FwdFeatSel(features_avail, X, TrainFcn, ExeFcn, LossFcn, outarg, ErrorTol, Kouter, Kinner)
    [features, StoppingCriteria, Egen_list, Etests, feat_hist, Etrain] = FeatSel(features_avail, 'fwd', X, Train, Exe, L, outarg, errortolerance, outer_train_cell, inner_train_cell);
toc %measure time

par_best = Train(X, features, 1);


%% plotting

genErr = figure('Name', 'Generalization Error', 'Position', [100 100 600 400], 'visible', 'off');
plot(Egen_list, '-', 'LineWidth', 1.5)
grid on
xticks(1:length(Egen_list))
xticklabels(feat_hist)
xlabel('Added features')
ylabel('Normalized euclidian distance')
title('Generalization Error')
legend({'Best Performing Model'})
saveas(genErr, 'Plots/LinReg_genErr.eps', 'epsc')

gen_trainErr = figure('Name', 'Test/Train Errors', 'Position', [100 100 600 400], 'visible', 'off');
hold on
    plot(sum(Etests,2)/Kouter, '-', 'LineWidth', 1.5)
    plot(sum(Etrain,2)/Kouter, '--', 'LineWidth', 1.5)
hold off
grid on
xticks(1:length(Egen_list))
xticklabels(feat_hist)
legend({'Generalization Error', 'Avg of Training Errors'})
xlabel('Added features')
ylabel('Normalized euclidian distance')
title('Outer layer Test/Training Errors during fwd feature selection')
saveas(gen_trainErr, 'Plots/LinReg_gen_trainErr.eps', 'epsc')

testErr = figure('Name', 'Test Errors', 'Position', [100 100 600 400], 'visible', 'off');
hold on
    plot(Etests(:,1), '-', 'LineWidth', 1.5)
    plot(Etests(:,2), '--', 'LineWidth', 1.5)
    plot(Etests(:,3), ':', 'LineWidth', 1.5)
    plot(Etests(:,4), '-.', 'LineWidth', 1.5)
    plot(Etests(:,5), '-', 'LineWidth', 3)
hold off
grid on
xticks(1:length(Egen_list))
xticklabels(feat_hist)
legend({'Outer Fold 1', 'Outer Fold 2', 'Outer Fold 3', 'Outer Fold 4', 'Outer Fold 5'})
xlabel('Added features')
ylabel('Normalized euclidian distance')
title('Test errors during fwd feature selection (best model of each split)')
saveas(testErr, 'Plots/LinReg_testErr.eps', 'epsc')


%% Print stopping reason
disp(' ')
disp(StoppingCriteria)
disp(' ')



%% Evaluate the best best model

P = cell(1);
M = cell(1);
    
P{1} = @(X)      LinRegTrain  (X, 1, features, outarg);
M{1} = @(par, X) LinRegExecute(par, X, features, outarg);

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

