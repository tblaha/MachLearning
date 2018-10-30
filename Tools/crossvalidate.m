function [Egen, s_select, Etest] = crossvalidate(X, P, M, L, outarg, Kouter, Kinner, seed)
% CROSSVALIDATE Takes models, data, output attr and loss function and
% selects to best model using K-fold validation. Also returns E_gen
% estimate
%   --- Inputs ---
%   X: data matrix with complete observations including output attribute
%   P: cell array of training functions such that:
%         par = P{i}(X) returns a struct of  trained parameters that y =
%         M{i}(par,X) needs to compute what the model thinks is the right
%         output argument y
%   M: cell array of model execution function such that:
%         y = M{i}(par, X) returns an vector of output attributes y
%         according to the model. NOTE: X is the truncated observation
%         matrix, so it DOES NOT contain the output attribute (of course)
%   L: loss function such that:
%         distance = L(y, yM) returns the distance between the observation
%         output vector y and what the model thinks is the vector of
%         outputs yM. Could be for example euclidian or cityblock or so
%   outarg: index of the output argument in the data matrix X
%   Kouter: number of outer cross validation folds
%   Kinner: number of inner cross validation folds
%   --- Outputs ---
%   Egen: Generalisation error estimate of the best models on the outer
%         test sets
%   s_select: vector containing the best model of each outer fold

    % if this function is called multiple times during for example fwd
    % features selection, then the splits must be the same for a fair
    % evaluation of the features.
    rng(seed)

    % preallocate some matrices
    Egen_models  = zeros(1,length(M));
    Etest        = zeros(1,Kouter);
    s_select     = zeros(1,Kouter);


    % divide targets: CV is object and contains indeces rather than actual
    % values
    CVouter = cvpartition(X(:,1), 'Kfold', Kouter);

    for i = 1:Kouter
        % outer training set --> inner set
        Xinner = X(CVouter.training(i),:);

        % outer test set
        Xouter_test = X(CVouter.test(i),:);


        % partition inner dataset
        CVinner = cvpartition(Xinner(:,1), 'Kfold', Kinner);

        % initialize matrices
        Eval = zeros(length(M), Kinner);
        for j = 1:Kinner 
            % training and test set
            Xinner_train = Xinner(CVinner.training(j),:);
            Xinner_test = Xinner(CVinner.test(j),:);

            for s = 1:length(M)

                Eval(s,j) = train_evaluate(Xinner_train, Xinner_test, outarg, M{s}, P{s}, L);

            end

        end

        for s = 1:length(M)

            % for each s, compute estimate of generalisation error
            Egen_models(s) = sum(CVinner.TestSize ./ CVouter.TrainSize(i) .*...
                          Eval (s,:));

        end

        % select best model
        [~, s_select(i)] = min(Egen_models); 

        % optimal model outer cross val test error
        Etest(i) = train_evaluate(Xinner, Xouter_test, outarg, M{s_select(i)}, P{s_select(i)}, L);

    end
    % estimate generalisation error as mean of all the outer cross val test
    % errors
    Egen = sum(CVouter.TestSize ./ length(X(:,1)) .*...
                          Etest);

    if ~all(s_select == s_select(1))
        disp("   Multiple different models were selected for different outer cross val folds")
        disp(mat2str(s_select))
    end


end

%% train and evaluate subroutine

function E = train_evaluate(Xtrain, Xtest, outarg, M, P, L)

    %%% training
    % train model to get parameters
    parameters = P(Xtrain);
    
    %%% evaluation
    % evaluate model
    o = 1:length(Xtest(1,:));
    o = o(~ismember(o,outarg)); % do not pass the output argument to the model.
    yM = M(parameters, Xtest(:,o));

    % validation error, invoke loss function with the output
    % arguments of the inner test set and the model output
    E = L(Xtest(:,outarg), yM);

end