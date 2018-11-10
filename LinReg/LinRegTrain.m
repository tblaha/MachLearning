function par = LinRegTrain(X, degree, features, outarg)
      
    % basis functions, simple linear with one constant term for now
    switch degree
        case 1
            par.vdm = @(X,lengthX) [ones(lengthX, 1), X];
            %par.vdm = @(X,lengthX) X;
            
        case 2
            par.vdm = @(X,lengthX) [ones(lengthX, 1), X, X.^2];
            
        case -1
            par.vdm = @(X,lengthX) [ones(lengthX, 1), 1./X];
            
    end
    
    % select output
    y    = X(:,outarg);
    
    % select inputs
    features = features( features ~= outarg ); % extra safety...
    
    X_pass = X(:,features);
    [a,~]  = size(X_pass);
    Xbar   = par.vdm(X_pass, a);
    
    % do the fitting using fwd feature selection
    par.w = (Xbar'*Xbar)\Xbar' * y;
    
end