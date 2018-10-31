function [yM, p] = NaiveBayesExecute(par, X, features, outarg)

    features(features > outarg) = features(features > outarg) - 1;
    
    %% calculating which values are bigger and smaller than mean - making binary matrices
    
    
    tl = size(X,1);  %test (set) length
    
    nbX_test = zeros(tl, length(features));     %test observations set to zeros, without output
    for k = 1:tl
        for j = features
            if (X(k, j) > par.mean_x_training(j))
                nbX_test(k,j) = 1;
            end
        end
    end
    
    %% p(x= x_test | y=1,2 or 3)

    p_xtest_y123 = ones(tl,3);  %initializing probabilities to 1

    for k = 1:tl
        for j = 1:3
            for i = features
                if (nbX_test(k,i) == 0)
                    p_xtest_y123(k,j) = p_xtest_y123(k,j) * par.p_x0_y(i,j);
                else
                    p_xtest_y123(k,j) = p_xtest_y123(k,j) * par.p_x1_y(i,j);
                end
            end
        end
    end

    %% final probabilities  p(y=1,2 or 3| x=x_test)

    p = zeros(tl,3);

    for k = 1:tl
        for j = 1:3
            p(k,j) = (p_xtest_y123(k,j)*par.p_y(j)) ...
                     / (   p_xtest_y123(k,1)*par.p_y(1) ...
                         + p_xtest_y123(k,2)*par.p_y(2) ...
                         + p_xtest_y123(k,3)*par.p_y(3) ); %bayes formula
        end
    end
    p
    
    %% output
    
    [~,yM] = max(p, [], 2); % index of max in all the rows
    

end