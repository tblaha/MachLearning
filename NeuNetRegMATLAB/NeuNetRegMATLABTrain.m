function par = NeuNetRegMATLABTrain(X, hiddenlayers, features, outarg)
      
    % select output
    y    = X(:,outarg);
    
    a = size(X,2);
    
    % select inputs
    features = features( features ~= outarg ); % extra safety...
    X = X(:,features);

    % random seed
    rng(2)
    
    % configure net
    net = fitnet(hiddenlayers);
    net.trainParam.showWindow  = 0;
    net.divideParam.testRatio  = 0;
    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio   = 0;
    
    % train
    [par, ~] = train(net, X', y');


    
end





