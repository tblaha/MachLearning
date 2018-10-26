function [Egen, s_select] = crossvalidate(X, P, M, L, outarg)



%% configuration

% number of cross validation folds
Kouter = 5;
Kinner = 10;



%% do magic

% preallocate some matrices
Egen_models  = zeros(1,length(M));
Etest        = zeros(1,Kouter);
s_select     = zeros(1,Kouter);


% divide targets: CV is object and contains indeces rather than actual
% values
CVouter = cvpartition(X(:,1), 'Kfold', Kouter); % 10 fold outer cross val

for i = 1:Kouter
    % outer training set --> inner set
    Xinner = X(CVouter.training(Kouter),:);
    
    % outer test set
    Xouter_test = X(CVouter.test(Kouter),:);
    
    
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
end


end

%% train and evaluate subroutine

function E = train_evaluate(Xtrain, Xtest, outarg, M, P, L)

    %%% training
    % train model to get parameters
    parameters = P(Xtrain);

    %disp(parameters);
    
    %%% evaluation
    % evaluate model
    o = 1:length(Xtest(1,:));
    o = o(o~=outarg); % do not pass the output argument to the model.
    yM = M(parameters, Xtest(:,o));

    % validation error, invoke loss function with the output
    % arguments of the inner test set and the model output
    E = L(Xtest(:,outarg), yM);

end


%% example models

function y = example_model(par, X)

    y = 1;

end

function parameters = example_model_train(X)

    parameters = struct;

end