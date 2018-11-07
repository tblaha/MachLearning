function par = BaselineRegTrain(X, outarg)
    par.mean = mean(X(:,outarg));
end