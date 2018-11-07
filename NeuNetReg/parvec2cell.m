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