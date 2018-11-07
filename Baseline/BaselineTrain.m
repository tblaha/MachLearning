function par = BaselineTrain(X, outarg)
    par.mode = mode(X(:,outarg));
end