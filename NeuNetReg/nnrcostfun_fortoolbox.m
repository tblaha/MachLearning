function c = nnrcostfun_fortoolbox(wvec, parin)

    par,p = parvec2cell(parin.layers, wvec);
    yM = NeuNetRegExecute(parin, parin.X, parin.features, parin.outarg);
    c = 1/length(parin.y) * sum(abs(parin.y-yM).^2); % euclidian
    
end