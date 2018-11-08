function par = NeuNetRegTrain(X, hiddenlayers, features, outarg)
      
    % select output
    y    = X(:,outarg);
    
    a = size(X,2);
    
    % select inputs
    features = features( features ~= outarg ); % extra safety...
    X = X(:,features);
    
    
    % form layer row vector
    par.layers = [length(features), hiddenlayers, length(outarg)];
    
    % length of parameter vector
    numpars = 0;
    for i = 1:length(par.layers)-1
        numpars = numpars + par.layers(i) * par.layers(i+1); % weights
    end
    numpars = numpars + sum(par.layers(2:end)); % biases
    
    % transfer function, should be replaced by smth like softmax in the
    % future
    par.transfun = @(z) tanh(z);
    
    % init weight/bias vector
    %wvec0 = fliplr([ zeros(1, sum(par.layers(2:end)) ), ones(1,5), -ones(1,numpars-sum(par.layers(2:end))-5)])';
    rng(2)
    wvec0 = rand(numpars,1);
    
    % train
    %%% using toolbox (requires backprop gradient, so not finished) 
    %par_fortoolbox = par;
    %par_fortoolbox.X = X;
    %par_fortoolbox.y = y;
    %par_fortoolbox.features = features;
    %par_fortoolbox.outarg = outarg;
    %opts = [1  1e-4  1e-8  1000];
    %[wvecopt, ~] = ucminf('nnrcostfun_fortoolbox', par_fortoolbox, wvec0, opts);
    
    %%% using internal matlab
    opts = optimoptions('fminunc', ...
                        'Display', 'off',...
                        'OptimalityTolerance', 3e-4, ...
                        'UseParallel', true, ...
                        'MaxFunctionEvaluations', 5000,...
                        'FiniteDifferenceType', 'forward');%,...
                        %'PlotFcn', {@optimplotfval});
    wvecopt = fminunc(@(wvec) nnrcostfun(wvec, par, X, y, features, outarg), wvec0, opts);
    
    
    %%% using internal matlab
    %opts = optimoptions('fmincon', ...
    %                    'Display', 'off',...
    %                    'OptimalityTolerance', 3e-4, ...
    %                    'UseParallel', true, ...
    %                    'MaxFunctionEvaluations', 5000,...
    %                    'PlotFcn', {@optimplotfval});
    %wvecopt = fmincon(@(wvec) nnrcostfun(wvec, par, X, y, features, outarg), wvec0, [], [], [], [], [], [], [], opts);
    
    % collect final parameters
    par.p = parvec2cell(par.layers, wvecopt);
    
end





