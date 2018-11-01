function par = NaiveBayesTrain(X, features, outarg) % requires non-one-out-of-k

    % extra safety
    features = features( ~ismember(features, outarg) );
    
    % pre alloc
    numf = length(features);
    par.mean_x_training = zeros(numf, 1);
    par.p_x0_y = zeros(numf,3);
    par.p_x1_y = zeros(numf,3);
    par.p_y = zeros(3,1);
    
    
    %% calculating which values are bigger and smaller than mean - making binary matrices

    par.mean_x_training = mean(X);
    tl=size(X,1);  %training (set) length
    nbX=zeros(tl,numf + length(outarg));    %training set initialized with all zeros

    for i=1:tl
        for j=features
            if ( X(i,j) > par.mean_x_training(j) )
                nbX(i,j) = 1;
            end
        end
    end
    
    nbX(:,outarg) = X(:,outarg);
    
    nbX
    %% p(y=1,2,3) -- priors

    par.p_y(1)=(sum(nbX(:,8)==1)+1)/(tl+3);  % +1 and +3 is for robustness
    par.p_y(2)=(sum(nbX(:,8)==2)+1)/(tl+3);
    par.p_y(3)=(sum(nbX(:,8)==3)+1)/(tl+3);
    
    %par.p_y(1) = 1/3;
    %par.p_y(2) = 1/3;
    %par.p_y(3) = 1/3;
    
    %% p(x_i=0|y=j)

    for j = 1:3
        for i = features
            par.p_x0_y(i,j)=sum((nbX(:,i)==0) & (nbX(:,outarg)==j))/sum(nbX(:,outarg)==j);
        end
    end
    
    %% p(x_i=1|y=j)
    
    par.p_x1_y = 1 - par.p_x0_y;
    
    
end