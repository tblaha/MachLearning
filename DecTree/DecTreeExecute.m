function yM = DecTreeExecute(par, X, features, outarg)

    for arg = sort(outarg, 'desc')
        features(features > arg) = features(features > arg) - 1;
    end

    label = predict(par, X(:, features));
    
    classNames = {'USA', 'Europe', 'Asia'}';
    yM = zeros(length(label), 1);
    for i = 1:length(label)
        yM(i) = find( strcmp( classNames, label(i) ) );
    end

end