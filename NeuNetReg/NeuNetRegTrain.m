function par = NeuNetRegTrain(X, hiddenlayers, features, outarg)
      
    % select output
    y    = X(:,outarg);
    
    a = size(X,2);
    
    % select inputs
    features = features( features ~= outarg ); % extra safety...
    X = X(:,~ismember(1:a,outarg));
    
    
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
    
    % train
    %wvec0 = fliplr([ zeros(1, sum(par.layers(2:end)) ), ones(1,5), -ones(1,numpars-sum(par.layers(2:end))-5)])';
    wvec0 = rand(numpars,1);
    
    opts = optimoptions('fminunc', ...
                        'Display', 'off',...
                        'OptimalityTolerance', 3e-4, ...
                        'UseParallel', true, ...
                        'MaxFunctionEvaluations', 5000,...
                        'PlotFcn', {@optimplotfval});
    wvecopt = fminunc(@(wvec) costfun(wvec, par, X, y, features, outarg), wvec0, opts);
    
    par.p = parvec2cell(par.layers, wvecopt);
    
end

function p = parvec2cell(layers, wvec)

    p.w = cell(size(layers, 2)-1, 1);
    p.b = cell(size(layers, 2)-1, 1);
    m = 1;
    
    % weights
    for i = 1:size(layers, 2)-1
        p.w{i} = zeros(layers(i), layers(i+1));
        for j = 1:layers(i+1)
            for k = 1:layers(i)
                p.w{i}(k,j) = wvec(m);
                m = m + 1;
            end
        end
    end
    
    % biases
    for i = 1:size(layers,2)-1
        for j = 1:layers(i+1)
            p.b{i}(j) = wvec(m);
            m = m + 1;
        end
    end

end


function c = costfun(wvec, par, X, y, features, outarg)

    par.p = parvec2cell(par.layers, wvec);
    yM = NeuNetRegExecute(par, X, features, outarg);
    c = 1/length(y) * sum(abs(y-yM).^2); % euclidian
    
end
