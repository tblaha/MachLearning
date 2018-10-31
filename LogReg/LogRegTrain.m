function par = LogRegTrain(X, features, outarg)
    
    % feature transformation
    par.X_trans = @(X, lengthX) [ones(lengthX,1), X];

    % basis functions, simple linear with one constant term for now
    par.sigm = @(X, w, lengthX) 1 ./ (1+exp(- par.X_trans(X, lengthX) * w));
    
    % cost function
    cost_func = @(X,w,y,lengthX) - sum ( y .* log( par.sigm(X, w, lengthX) ) + (1-y) .* log( 1 - par.sigm(X, w, lengthX) ) ); 
    
    
    
        
    % select inputs
    features = features( ~ismember(features, outarg) ); % extra safety...
    X_pass = X(:,features);
        
    [a,~]  = size(X_pass);
    [~, b] = size( par.X_trans(X_pass, a) );
    par.w = zeros(b, length(outarg));
    
    % do stuff
    for i = 1:length(outarg)
        % select output
        y = X(:,outarg(i));
        
        % do the fitting using numerical optimization
        w0 = zeros(b,1); % w "zero" not "oh"
        options = optimoptions('fmincon','Display','off');
        par.w(:,i) = fmincon(@(w) cost_func(X_pass, w, y, a), w0, [], [], [],[],[],[], [], options); 
    end
    
end