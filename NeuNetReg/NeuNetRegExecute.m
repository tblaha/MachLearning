function yM = NeuNetRegExecute(par, X, features, outarg)

    % feature vector was meant for the unmodified X. But now we get only
    % the truncated X (with the output feature removed), so we need to
    % offset the feature vector by -1 for all features with id bigger than
    % outarg.
    features(features > outarg) = features(features > outarg) - 1;
    X = X(:, features);
    
    [a,~] = size(X(:, features));
    
    yM = zeros(a,1);
    for i = 1:a    % for each data point
        activations = {X(i,:)'};
        for j = 1:length(par.layers)-1 % for each layer
            activations{j+1} = link(j, activations, par.layers, par.wc, par.transfun);
        end
        yM(i) = activations{end}; 
    end

end

function out = link(j, activations, layers, wc, transfun)

    numlay = size(layers, 2);
    
    out = zeros(layers(j+1),1);
    for k = 1:layers(j+1) % for each next neuron        
        for m = 1:layers(j) % for current neuron
            out(k) = out(k) ... 
                   + transfun(activations{j}(m), wc{j}(m,k), max(0, 2 + j - numlay)); % next activation
        end
    end
    
end