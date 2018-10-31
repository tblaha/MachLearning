function yM = LogRegExecute(par, X, features, outarg)

    % feature vector was meant for the unmodified X. But now we get only
    % the truncated X (with the output feature removed), so we need to
    % offset the feature vector by -1 for all features with id bigger than
    % outarg.
    for arg = sort(outarg, 'desc')
        features(features > arg) = features(features > arg) - 1;
    end
            
    [a,~] = size(X(:,features));
    activation = par.sigm(X(:,features),par.w, a); % par.sigm contains the feature transformations
    
    yMmat = (activation > 1/2);
    
    % code back to original 1, 2, 3 coding
    yM = zeros(a,1);
    if length(outarg) == 2
        for i = 1:a
            if xor(yMmat(i,1), yMmat(i,2)) % if both are different
                yM(i) = find(yMmat(i,:)); % assign 1 if it is the first (USA), 2 if it is the second (Europe)
            elseif ~and(yMmat(i,1), yMmat(i,2)) % if both are 0
                yM(i) = 3; % Japan
            else % both are 1
                [~,yM(i)] = max(activation(i,:)); % assign the index of whichever activation is higher.
            end
        end
    elseif length(outarg) == 3
        
        for i = 1:a
            [~, yM(i)] = max(activation(i,:));
        end
        
    end

end