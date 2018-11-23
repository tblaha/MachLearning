function [logli_gen, s_select, logli_test, logli_train] = crossvalidateClass(PCAs, P, M, outercell, innercell)

    % size of splits
    Kouter = length(outercell);
    Kinner = size(innercell{1},2);
    
    % preallocate some matrices
    logli_test   = zeros(1,Kouter);
    logli_train  = zeros(1,Kouter);
    s_select     = zeros(1,Kouter);

    % counter
    cnt = 0;
    
    for i = 1:Kouter
        logli_gen_models  = zeros(1,length(M));
        
        % outer training set --> inner set
        Xinner = PCAs(outercell{i}, :);

        % outer test set
        Xouter_test = PCAs(~outercell{i}, :);

        % initialize matrices
        Eval = zeros(length(M), Kinner);
        for j = 1:Kinner 
            % training and test set
            Xinner_train = Xinner( innercell{i}(:,j),:);
            Xinner_test  = Xinner(~innercell{i}(:,j),:);

            for s = 1:length(M)
                Eval(s,j) = train_evaluate(Xinner_train, Xinner_test, M{s}, P{s});
            end
            % statusupdate
            cnt = cnt + 1;
            disp(cnt / (Kouter*Kinner) * 100)
            

        end

        for s = 1:length(M)

            % for each s, compute estimate of generalisation error
            logli_gen_models(s) = sum(sum(~innercell{i},1) ./ sum(outercell{i}) .*...
                                  Eval (s,:));

        end

        % select best model
        [~, s_select(i)] = min(logli_gen_models); 

        % optimal model outer cross val test error
        [logli_test(i), logli_train(i)] = train_evaluate(Xinner, Xouter_test, M{s_select(i)}, P{s_select(i)});
        
                
    end
    % estimate generalisation error as mean of all the outer cross val test
    % errors
    logli_gen = sum( sum(~outercell{i}) ./ length(outercell{i}) .*...
                          logli_test);

                      
    if ~all(s_select == s_select(1))
        disp("   Multiple different models were selected for different outer cross val folds")
        disp(mat2str(s_select))
    end


end

%% train and evaluate subroutine

function [Etest, Etrain] = train_evaluate(Xtrain, Xtest, M, P)

    %%% training
    % train model to get parameters
    try
        parameters = P(Xtrain);
    
        Etrain = M(parameters, Xtrain);
         
        % test set
        Etest = M(parameters, Xtest);
    catch ME
        % gmm is full of shit
        Etest = 1e10;
    end
end