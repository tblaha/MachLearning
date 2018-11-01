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
        numpars = numpars + par.layers(i) * par.layers(i+1);
    end
    
    % transfer function, should be replaced by smth like softmax in the
    % future
    par.transfun = @(z) tanh(z);
    
    % train
    wvec0 = [ones(3,1); -ones(numpars-3,1)] ;
                        %'Display', 'notify-detailed',...
    opts = optimoptions('fminunc', ...
                        'Display', 'off',...
                        'OptimalityTolerance', 3e-4, ...
                        'UseParallel', true, ...
                        'MaxFunctionEvaluations', 5000,...
                        'PlotFcn', {@optimplotfval});
    %opts = optimoptions('fminunc', 'Display', 'notify-detailed', 'OptimalityTolerance', 5e-4);

    wvecopt = fminunc(@(wvec) costfun(wvec, par, X, y, features, outarg), wvec0, opts);
        
    par.wc = parvec2cell(par.layers, wvecopt);
    
end

function wc = parvec2cell(layers, wvec)

    wc = cell(size(layers, 2)-1);
    m = 1;
    for i = 1:size(layers, 2)-1
        wc{i} = zeros(layers(i), layers(i+1));
        for j = 1:layers(i+1)
            for k = 1:layers(i)
                wc{i}(k,j) = wvec(m);
                m = m + 1;
            end
        end
    end

end


function c = costfun(wvec, par, X, y, features, outarg)

    par.wc = parvec2cell(par.layers, wvec);
    yM = NeuNetRegExecute(par, X, features, outarg);
    c = 1/length(y) * sum(abs(y-yM).^2); % euclidian
    
end
