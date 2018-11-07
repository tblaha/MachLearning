function yM = BaselineRegExecute(par, X)
    yM = par.mean .* ones(size(X,1),1);
end