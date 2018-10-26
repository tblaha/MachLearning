function yM = LinRegExecute(par, X, features, outarg)

    % feature vector was meant for the unmodified X. But now we get only
    % the truncated X (with the output feature removed), so we need to
    % offset the feature vector by -1 for all features with id bigger than
    % outarg.
    features(features > outarg) = features(features > outarg) - 1;
    
    [a,~] = size(X(:, features));
    yM = par.vdm(X(:, features), a) * par.w;

end