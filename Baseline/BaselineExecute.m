function yM = BaselineExecute(par, X)
    yM = par.mode .* ones(size(X,1),1);
end