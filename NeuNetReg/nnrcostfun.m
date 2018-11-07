function c = nnrcostfun(wvec, par, X, y, features, outarg)

    par.p = parvec2cell(par.layers, wvec);
    yM = NeuNetRegExecute(par, X, features, outarg);
    c = 1/length(y) * sum(abs(y-yM).^2); % euclidian
    
end