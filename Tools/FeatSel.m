function [features, stoppingCriteria, Egen_list, Etests] = FeatSel(features_avail, fwdbwd, X, TrainFcn, ExeFcn, LossFcn, outarg, ErrorTol, outer_train_cell, inner_train_cell)
%FWD_FEAT_SEL Performs fwd features selection using 2-layer crossval
%   --- Inputs ---
%   features_avail: vector of indices in X(:,index) corresponding to all
%                   features that can be tried during the selection
%   X: data matrix with complete observations including output attribute
%   TrainFcn: template of the training function such that:
%         par = TrainFcn(X, features, outarg) returns a struct of the
%         trained paramters, given a data matrix X (containing the output
%         argument), the features to use and the index of the output attrib
%   ExeFcn: template of the model execution function such that:
%         yM = ExeFcn(par, X, features, outarg) returns what the model
%         thinks are the correct outputs, given the trained paramters, the
%         truncated data matrix X (column with output attributes removed),
%         the features that were used during training and the index of the 
%         output attribute in the full data matrix X. Note that the
%         truncation of X takes place in the crossvalidate.m function
%   LossFcn: loss function such that:
%         distance = L(y, yM) returns the distance between the observation
%         output vector y and what the model thinks is the vector of
%         outputs yM. Could be for example euclidian or cityblock or so
%   outarg: index of the output argument in the full data matrix X
%   ErrorTol: when to about the feature selection because the using another
%             feature only decreases the generalization error by less than
%             ErrorTol
%   Kouter: number of outer cross validation folds
%   Kinner: number of inner cross validation folds
%   seed: random seed to use for the splits when comparing models using
%         crossvalidation
%
%   --- Outputs ---
%   features: the features selected
%   stoppingCriteria: string of why the iterations stopped
%   Egen_list: time history of generalization error estimates
    
    current_Egen = 0;
    if strcmp(fwdbwd, 'fwd')
        features = [];
    else
        features = features_avail;
    end
    
    j = 2;
    P{1} = @(X)      TrainFcn  (X,      1:7, outarg);
    M{1} = @(par, X) ExeFcn    (par, X, 1:7, outarg);
    [ Egen_list(1), ~, Etests(1,:)] = crossvalidate(X, P, M, LossFcn, outarg, outer_train_cell, inner_train_cell);

    
    while 1
        
        previous_Egen = current_Egen;

        % initiate
        P = cell(length(features_avail) + 1, 1);
        M = cell(length(features_avail) + 1, 1);

        % configure baseline of the privous iteration
        P{1} = @(X)      TrainFcn  (X,      features, outarg);
        M{1} = @(par, X) ExeFcn    (par, X, features, outarg);

        % iterate over the next fwd features to try
        i = 2;
        for new_feat = features_avail
            if strcmp(fwdbwd, 'fwd')
                features_try = sort([features, new_feat]);
            else
                features_try = features(features ~= new_feat);
            end

            P{i} = @(X)      TrainFcn  (X, features_try, outarg);
            M{i} = @(par, X) ExeFcn    (par, X, features_try, outarg);

            i = i + 1;
        end

        % see which of the models is best
        [~, s_select, Etests(j,:)] = crossvalidate(X, P, M, LossFcn, outarg, outer_train_cell, inner_train_cell);
        
        % little hack to make it prefer larger model numbers (ie not the
        % baseline in case that more models have an equal amount of splits
        % where they were best.
        s_select_inv = 1000 - s_select;
        s_best = mode(s_select_inv);
        s_best = 1000 - s_best;
        
        % find generalization error for that model
        current_Egen = crossvalidate(X, P(s_best), M(s_best), LossFcn, outarg, outer_train_cell, inner_train_cell);
        
        % add to array
        Egen_list(j) = current_Egen;
        disp(num2str(j))
        j = j + 1;


        %if current_Egen < previous_Egen % so the baseline of the previous iteration is better than any other model
        if s_best == 1
            stoppingCriteria = "Broken by baseline being better, nice.";
            break;
        end
        if abs(current_Egen - previous_Egen) < ErrorTol
            stoppingCriteria = "Broken by reaching error tolerance of the generalization error, nice.";
            break;
        end
        

        % we have survived the stopping criteria, we seem to have a new
        % features, see which one it is:
        best_feature = features_avail(s_best - 1);
        disp(best_feature)

        % new base line is:
        if strcmp(fwdbwd, 'fwd')
            features = sort([features, best_feature]);
        else
            features = features(features ~= best_feature);
        end

        % new available features are
        features_avail = features_avail( features_avail ~= best_feature );
        disp(features)
        
        if isempty(features_avail) % no more features left to try
            stoppingCriteria = "Broken by not reaching the error tolerance of the generalization error. All features selected. Not so nice.";
            break;
        end

    end

end