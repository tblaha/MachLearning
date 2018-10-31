function par = DecTreeTrain(X, features, outarg, minpar)
    
    %% Preparing
    
    % selecting outputs
    classNames= {'USA', 'Europe', 'Asia'}';
    y=zeros(size(X,1),1);

    if(size(X,2)>8)     %If X one out of K is runned, else it's normal X
        for i = 1:size(X,1)
            if(X(i,8)==1)
                y(i)=1;
            elseif(X(i,9)==1)
                y(i)=2;
            elseif(X(i,10)==1)
                y(i)=3;
            end
        end
        for i=1:4   %Removing origins and year
            X(:,7)=[];
        end
    else
       y=X(:,outarg);    %Saving the origins to be used for classNames
    end
    
    % select inputs
    attributeNames = {'gpm', 'cylinders', 'displacement', 'horsepower', 'weight', 'accelleration'};
    features = features( ~ismember(features, outarg) ); % extra safety...
    X = X(:,features);

    %% outlier detection (needs to be re-written for the new standardized data)
    
    %Boxplot to find outliers
    %boxplot(X, attributeNames, 'LabelOrientation', 'inline');
    %boxplot(zscore(X), attributeNames, 'LabelOrientation', 'inline');

    %Removing outliers from data set
    %idxOutlier = find(X(:,4)>200 | X(:,6)>21.9 | X(:,6)<9 | X(:,1)>0.09);
    %X(idxOutlier,:) = [];
    %y(idxOutlier) = [];

    %% Fit the tree
    
    par = fitctree(X, classNames(y), ...
       'splitcriterion', 'gdi', ...
       'categorical', [], ...
       'PredictorNames', attributeNames, ...
       'prune', 'off', ...
       'minparent', minpar);



end