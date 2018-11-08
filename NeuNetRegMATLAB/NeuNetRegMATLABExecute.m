function yM = NeuNetRegMATLABExecute(par, X, features, outarg)

    % feature vector was meant for the unmodified X. But now we get only
    % the truncated X (with the output feature removed), so we need to
    % offset the feature vector by -1 for all features with id bigger than
    % outarg.
    features(features > outarg) = features(features > outarg) - 1;
    X = X(:, features);
    
    a = size(X,1);
    
    % evaluate
    yM = sim(par.net, X')';
    

end