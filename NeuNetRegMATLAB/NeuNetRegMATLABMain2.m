addpath(genpath('../'))
warning('off', 'all')

clear

%% Configuration

% import data
importdata_Report2 % non-one-out-of-k-coded

% p-norm:
p_dist = @(y,yM,p) 1/length(y) * sum(abs(y-yM).^p);

% Loss function:
% y is test data output, yM is what the model thinks it is
L = @(y,yM) p_dist(y, yM, 2);   % euclidian  p = 2
%L = @(y,yM) p_dist(y, yM, 1);   % city block p = 1


% complexity control parameters
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
features = features_avail( features_avail ~= outarg );
features = [2 3 4 5 6 7 8];

% don't do feature selection, just take all. Instead, do complexity control
hmax = 14;
hes = 2:hmax;
%hes = 0;


for i = 1:length(hes) % least complex to most complex
    hiddenlayers = hl_try(hes(i));
    Train = cell(1, 1);
    Exe   = cell(1, 1);
    Train{1} = @(     X) NeuNetRegMATLABTrain  (X,   hiddenlayers, features, outarg); % 1 stands for first order reg
    Exe{1}   = @(par, X) NeuNetRegMATLABExecute(par, X  , features, outarg);
    [Egen(i), s_select(i,:), Etest(i,:), Etrain(i,:)] = crossvalidate(X, Train, Exe, L, outarg, outer_train_cell, inner_train_cell);
    disp(i)
end

%tic
%[Egen, s_select, Etest, Etrain] = crossvalidate(X, Train, Exe, L, outarg, outer_train_cell, inner_train_cell);
%toc

[~,s_idx] = min(Egen);
Egen_best = Egen(s_idx);
idx = s_select(s_idx);

par_best = NeuNetRegMATLABTrain(X, hl_try(hes(idx)), features, outarg);

%% plotting

genvstrainErr = figure('Name', 'Training and Generalization Error', 'Position', [100 100 600 400], 'visible', 'on');
hold on
    plot(hes, Egen, '-', 'LineWidth', 1.5)
    plot(hes, mean(Etrain, 2), '--', 'LineWidth', 1.5)
hold off
xticks(hes)
ylim([0,0.5])
grid on
xlabel('Hidden Neurons')
ylabel('Normalized euclidian distance')
title('Generalization and Training Error')
legend({'Gen. Error Estimate', 'Mean of Training Error'}, 'Location', 'NorthWest')
saveas(genvstrainErr, 'Plots/ANN_gentrainErr.eps', 'epsc')


%% output


disp(' ')
disp('|----- Calculations finished -----|')
disp(' ')
disp(strcat('Selected number of hidden neurons: ', mat2str(hl_try(hes(idx)))))
disp(' ')
disp(strcat('Estimated generalisation error: ', num2str(Egen_best)))
disp(' ')

function hiddenlayers = hl_try(hlnum)

    if hlnum == 0
        hiddenlayers = [];
    elseif hlnum <= 8
        hiddenlayers = [hlnum];
    elseif hlnum > 8
        lastl = min(round(hlnum/2),8);
        hiddenlayers = [hlnum - lastl,lastl];
    end
        
end

