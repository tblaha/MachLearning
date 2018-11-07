function [outer_cell, inner_cell] = genSplits(X, Kouter, Kinner, seed)
    
    % set random seed
    rng(seed)
    
    % Save splits for all folds already now.
    % Save as column vectors for outer
    % Save as 3d matrix for inner
    CVouter = cvpartition(X(:,1), 'Kfold', Kouter);
    outer_cell = cell(Kouter, 1);
    inner_cell = cell(Kinner, 1);


    for a = 1:Kouter
        outer_cell{a} = logical(CVouter.training(a));

        X_outer_train = X(outer_cell{a}, :);
        inner_cell{a} = logical(zeros(size(X_outer_train, 1), Kinner));

        CVinner = cvpartition(X_outer_train(:,1), 'Kfold', Kinner);
        for b = 1:Kinner
            inner_cell{a}(:,b) = logical(CVinner.training(b));
        end
end
    
end