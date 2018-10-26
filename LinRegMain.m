importdata_Report1

addpath(genpath('./'))

% Loss function:
% y is test data output, yM is what the model thinks it is
%L = @(y,yM) 1/length(y) * sum((y-yM).^2);   % euclidian
L = @(y,yM) 1/length(y) * sum(abs((y-yM))); % city block

%% Fwd features selection

% which argument is output?
outarg = 1; % id of the X(:,id) data matrix. 1: gpm

% This means runing the crossvalidate function a lot of times with
% different models as input
features_avail = 1:length(X(1,:));
features_avail = features_avail( features_avail ~= outarg );

%fwd feature selection
previous_Egen = 1;
current_Egen = 0;
features = [];


tic %measure time
while 1
    previous_Egen = current_Egen;
    
    % initiate
    P = cell(length(features_avail) + 1, 1);
    M = cell(length(features_avail) + 1, 1);
       
    % configure baseline of the privous iteration
    P{1} = @(X)      LinRegTrain  (X, 1, features, outarg);
    M{1} = @(par, X) LinRegExecute(par, X, features, outarg);
    
    % iterate over the next fwd features to try
    i = 2;
    for new_feat = features_avail
        features_try = sort([features, new_feat]);
        
        P{i} = @(X)      LinRegTrain  (X, 1, features_try, outarg);
        M{i} = @(par, X) LinRegExecute(par, X, features_try, outarg);
        
        i = i + 1;
    end
    
    % see which of the models is best
    [current_Egen, s_select] = crossvalidate(X, P, M, L, outarg);
    s_best = mode(s_select);
        
    if s_best == 1 % so the baseline of the previous iteration is better than any other model
        disp(' ')
        disp("Broken by baseline being better, nice.")
        break;
    end
    if isempty(features_avail) % no more features left to try
        disp(' ')
        disp("Broken by not reaching the error tolerance of the generalization error. All features selected. Not so nice.")
        break;
    end
    if abs(current_Egen - previous_Egen) < 0.0001
        disp(' ')
        disp("Broken by reaching error tolerance of the generalization error, nice.")
        break;
    end
    
    % we have survived the stopping criteria, we seem to have a new
    % features, see which one it is:
    best_feature = features_avail(s_best - 1);
    
    % new base line is:
    features = sort([features, best_feature]);
    
    % new available features are
    features_avail = features_avail( features_avail ~= best_feature );
    
end
disp(' ')
toc %measure time






%% Evaluate the best best model

P = cell(1);
M = cell(1);
    
P{1} = @(X)      LinRegTrain  (X, 1, features, outarg);
M{1} = @(par, X) LinRegExecute(par, X, features, outarg);


[Egen, s_select] = crossvalidate(X, P, M, L, outarg);


%% output

disp(' ')
disp('|----- Calculations finished -----|')
disp(' ')
disp(strcat('Selected model features: ', mat2str(features)))
disp(' ')
disp(strcat('Estimated generalisation error: ', num2str(Egen)))
disp(' ')