importdata_Report2

%% configuration

% which argument is output?
outarg = 1; % id of the X(:,id) data matrix. 1: gpm

% number of cross validation folds
Kouter = 10;
Kinner = 5;

% Loss function:
% y is test data output, yM is what the model thinks it is
L = @(y,yM) 1/length(y) * sum((y-yM).^2); % euclidian

%% declare the models --> this must be modified.

% Train models
P1 = @(X) example_model_train(X);
P2 = @(X) example_model_train(X);
P3 = @(X) example_model_train(X);

P = {P1 P2 P3};


% Evaluate models
M1 = @(X, par) example_model(par, X);
M2 = @(X, par) example_model(par, X)*2;
M3 = @(X, par) example_model(par, X)/2;

M = {M1 M2 M3};



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
    disp("Multiple different models were selected for different outer cross val folds")
end


%% train and evaluate subroutine

function E = train_evaluate(Xtrain, Xtest, outarg, M, P, L)

    %%% training
    % train model to get parameters
    parameters = P(Xtrain);

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