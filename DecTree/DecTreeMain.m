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
L = @(y,yM) bayesloss(y,yM);


% model functions
% --> will be declared in for loop


% Tree level analysis configuration
seed = 2; % random seed used for crossval splits
errortolerance = 0.0001;

% cross validation configuration
Kouter = 5;
Kinner = 5;

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


splc = {'gdi', 'twoing', 'deviance'};

Egen_hist = [];
minpar_hist = ones(length(splc),1) .* 70; % the lower, the more complex the tree
alpha = 0.8; % decrease minpar by (roughly) (1-alpha) *100 percent every iteration
strikes = 10; % how many consecutive worse iterations to tolerate


for sc = 1:length(splc)
    i = 1;
    while true    
        % model functions
        Train = @(     X) DecTreeTrain  (     X, features, outarg, minpar_hist(sc,i), splc{sc}); % 1 stands for first order reg
        Exe   = @(par, X) DecTreeExecute(par, X, features, outarg);

        % check best model
        [Egen_hist(sc,i), ~] = crossvalidate(X, {Train}, {Exe}, L, outarg, outer_train_cell, inner_train_cell);

        % new minpar
        minpar_hist(sc, i+1) = round(minpar_hist(sc,i)*alpha);

        % status update
        disp(i)
        disp(Egen_hist(i))

        % stopping criterium
        if i > strikes && sum(Egen_hist(sc,i-strikes) - Egen_hist(sc,i-strikes+1:i) < errortolerance * ones(1,strikes)) == strikes
            % this means that for strikes iterations already no improvements 
            minpar(sc) = minpar_hist(sc,i-strikes);
            Egen(sc)   = Egen_hist(sc,i-strikes);
            break;
        end

        % counting
        i = i + 1;

    end
    
end
%%
minpar_temp=6;
  Train = @(     X) DecTreeTrain  (     X, features, outarg, minpar_temp, 'gdi'); % 1 stands for first order reg
       Exe   = @(par, X) DecTreeExecute(par, X, features, outarg);
 [~,~, Etest] = crossvalidate(X, {Train}, {Exe}, L, outarg, outer_train_cell, inner_train_cell);
%%
figure('Name', 'Generalization Error')
hold on
    plot(minpar_hist(1,1:end-5),Egen_hist(1,1:end-4), 'LineWidth', 1.5)
    plot(minpar_hist(2,1:end-5),Egen_hist(2,1:end-4),':', 'LineWidth', 1.5)
    plot(minpar_hist(3,1:end-5),Egen_hist(3,1:end-4),'--', 'LineWidth', 1.5)
legend('Ginis diversity index','Twoing rule', 'Cross entropy')
hold off
title('Generalization error of Decision tree')
xlabel('Parameter: Minimum parents')
ylabel('Generalization error')
grid on
set(gca, 'XDir','reverse')
saveas(gcf,'DecTree_genErr','epsc')

%%
% return tree
%outtree = DecTreeTrain  (X, features, outarg, minpar, splc);

%% output

disp(' ')
disp('|----- Calculations finished -----|')
for i = 1:length(splc)
    disp(strcat('Method: ', string(splc(i))))
    disp(strcat('Best minimum parents level: ', mat2str(minpar(i))))
    disp(' ')
    disp(strcat('Estimated generalisation error: ', num2str(Egen(i))))
    disp(' ')
end

%% logistic loss function

function loss = bayesloss(y,yM) % y is 8th and 9th column
    
    %ySingle = (y(:,1) == 1) *1 + (y(:,2) == 1) *2 + (y(:,1) == 0 & y(:,2) == 0) *3; % code back from one out of k
    loss = sum(abs((y - yM)) > 0) / length(y);

end
