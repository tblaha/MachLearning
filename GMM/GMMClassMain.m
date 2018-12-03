addpath(genpath('../'))
warning('off', 'all')

clear

PCA
%PCAs = X*Px(:,[1,2]);
PCAs = X*Px(:, [1,2,4]);

%% Configuration

% complexity control parameters
seed = 2; % random seed used for crossval splits

% cross validation configuration
Kouter = 10;
Kinner = 2;

% generate splits
[outer_train_cell, inner_train_cell] = genSplits(PCAs, Kouter, Kinner, seed);



%% Fwd features selection

% don't do feature selection, just take all. Instead, do complexity control
num_gau = 2:10;

BIC = nan(1, length(num_gau));
AIC = nan(1, length(num_gau));

parfor i = 1:length(num_gau) % least complex to most complex
    Train = cell(length(num_gau), 1);
    Exe   = cell(length(num_gau), 1);
    Train{1} = @(     X) GMMClassTrain  (X,   num_gau(i), seed); % 1 stands for first order reg
    Exe{1}   = @(par, X) GMMClassExe    (par, X);
    gmtemp  = Train{1}(PCAs);
    BIC(i)  = gmtemp.gm.BIC;
    AIC(i)  = gmtemp.gm.AIC;
    Egen(i) = crossvalidateClass(PCAs, Train, Exe, outer_train_cell, inner_train_cell);
end

% tic
%     [Egen, s_select, Etest, Etrain] = crossvalidateClass(PCAs, Train, Exe, outer_train_cell, inner_train_cell);
% toc
% idx = mode(s_select);

[~, idx] = min(Egen);
par_best = GMMClassTrain(PCAs, num_gau(idx), seed);


%%

h = figure('Position', [100 100 800 500], 'Visible', 'on');
title('Crossvalidation and Bayes Information Criterion for the Clustering', 'FontSize', 14)
xlabel('Clusters', 'FontSize', 14)
ylabel('Information', 'FontSize', 14)
hold on
    plot(num_gau, Egen*Kouter*Kinner, 'LineWidth', 1.5, 'LineStyle', '-', 'DisplayName', 'Crossvalidation')
    plot(num_gau, BIC(1:9), 'LineWidth', 1.5, 'LineStyle', '--', 'DisplayName', 'BIC')
    plot(num_gau, AIC(1:9), 'LineWidth', 1.5, 'LineStyle', '-.', 'DisplayName', 'AIC')
hold off
legend({}, 'FontSize', 14, 'Location', 'NorthWest')
grid on
saveas(h, 'Plots/GMM_Errors.eps', 'epsc')


%% output


disp(' ')
disp('|----- Calculations finished -----|')
disp(' ')
disp(strcat('Selected number of classes: ', num2str(num_gau(idx)) ))
disp(' ')
disp(strcat('Estimated generalisation error: ', num2str(Egen)))
disp(' ')



%% Data prep
 
cyls    = data.cylinders;
classes = X(:,8);
scat = PCAs;

cylgroups = {[3] [4] [5] [6] [8]};

origs = {"USA", "Europe", "Japan"};


cyl_classes = 1 * (ismember(cyls, cylgroups{1})) ...
            + 2 * (ismember(cyls, cylgroups{2})) ...
            + 3 * (ismember(cyls, cylgroups{3}));


%% Reference plots

% figCyl = figure('Position', [100 100 800 500], 'Visible', 'off');
% hold on
% for k = 1:size(cylgroups,2)
%     scatter(scat(ismember(cyls,cylgroups{k}),1),...
%             scat(ismember(cyls,cylgroups{k}),2),... 'b',...
%             'DisplayName', strcat('Cylinder: ', mat2str(cylgroups{k}')))
% end
% hold off
% title('Scatter Plot for the raw data -- Number of Cylinders', 'FontSize', 16)
% xlabel('PCA1 score', 'FontSize', 16)
% ylabel('PCA2 score', 'FontSize', 16)
% grid on
% axis equal
% hold off
% legend({}, 'Location', 'SouthWest', 'Fontsize', 14)
% saveas(figCyl, 'Plots/ScatCyl.eps', 'epsc')


figOri = figure('Position', [100 100 800 500], 'Visible', 'on');
hold on
for k = 1:length(origs)
    scatter3(scat(X(:,8) == k, 1),...
             scat(X(:,8) == k, 2),... 'b',...
             scat(X(:,8) == k, 3),...
            'DisplayName', strcat('Origin: ', origs{k}));
end
hold off
title('Scatter Plot for the raw data -- Origin', 'FontSize', 16)
xlabel('PCA1 score', 'FontSize', 16)
ylabel('PCA2 score', 'FontSize', 16)
zlabel('PCA4 score', 'FontSize', 16)
grid on
hold off
legend({}, 'Location', 'SouthWest', 'Fontsize', 14)
saveas(figOri, 'Plots/ScatOri.eps', 'epsc')


%% GMM
% 
% % select model
% num_gau_plot = 3;
% par_plot = GMMClassTrain(PCAs, num_gau_plot, seed);
% 
% gmPDF = @(x) pdf(par_plot.gm,x);
% 
% g = figure('Position', [100 100 800 500], 'Visible', 'on');
% hold on
% % ezsurf(@(x,y) gmPDF([x;y]'), [-10 10], [-10 10], 201)
% h = ezcontour(@(x,y) gmPDF([x;y]') ,[-6 6],[-2 4], 201);
% for k = 1:size(cylgroups,2)
% %     scatter(scat(ismember(cyls,cylgroups{k}),1),...
% %             scat(ismember(cyls,cylgroups{k}),2),... 'b',...
% %             'DisplayName', strcat('Cylinder: ', mat2str(cylgroups{k}')))
%     if k == 1
%         scatter(scat(ismember(cyls,cylgroups{k}),1),...
%              scat(ismember(cyls,cylgroups{k}),2), 'b',...
%              'DisplayName', 'Data')
%     else
%         scatter(scat(ismember(cyls,cylgroups{k}),1),...
%              scat(ismember(cyls,cylgroups{k}),2), 'b', 'HandleVisibility', 'off')
%     end
%     %ezcontour(@(x1,x2)pdf(gm,[x1 x2]))
% end
% for k = 1:num_gau_plot
%     scale = chi2inv(0.95,2);     %# inverse chi-squared with dof=#dimensions
%     sigma = par_plot.gm.Sigma(:,:,k);
%     [V, D] = eig(sigma);
%     
%     th = linspace(0, 2*pi, 101);
%     circ = [cos(th); sin(th)] .* sqrt(diag(D)) * sqrt(scale);
%     
%     x = par_plot.gm.mu(k,:)' + V * circ;
%     if k == 1
%         plot(x(1,:), x(2,:), 'k--', 'DisplayName', '95% Prediction')
%     else
%         plot(x(1,:), x(2,:), 'k--', 'HandleVisibility','off')
%     end
% end
% plot(par_plot.gm.mu(:,1), par_plot.gm.mu(:,2), 'kx', 'MarkerSize', 20, 'DisplayName', 'Cluster Center')
% title(sprintf('Scatter Plot, GMM Contour, 95%% confidence ellipses: %d Clusters', num_gau_plot), 'FontSize', 14)
% xlabel('PCA1 score', 'FontSize', 14)
% ylabel('PCA2 score', 'FontSize', 14)
% grid on
% axis equal
% hold off
% legend({'GMM levels', 'Observations', '95% Prediction', 'Cluster Centers'})
% saveas(g, sprintf('Plots/Cluster%d.eps', num_gau_plot), 'epsc')


%% Quality evaluation

cnt = 1;
for num_gau_eval = 3
    % -- origin --
    par_eval = GMMClassTrain(PCAs, num_gau_eval, seed);
    gmPDF = @(x) pdf(par_eval.gm,x);

    [~, most_likely_cluster] = max(posterior(par_eval.gm, PCAs),[],2);


    % try out all num_gau_plot!/(num_gau_plot - num_groups)! possible interpretations of the centres
    num_groups = 3;
    num_v = factorial(max(num_gau_eval, num_groups))/factorial(abs(num_gau_eval - num_groups));

    %permutes = perms([1:min(num_gau_plot, num_groups), zeros(1,abs(num_gau_plot - num_groups))]);
    %actual_permutes = unique(permutes,'rows');

    permutes = perms(1:max(num_gau_eval, num_groups));
    permutes(ismember(permutes, ( (min(num_groups,num_gau_eval)+1):max(num_gau_eval, num_groups) ) ) ) = -1;
    actual_permutes = unique(permutes, 'rows');

    correct = zeros(num_v, 1);
    i = 1;
    for v = actual_permutes'
        % assign groups
        if num_gau_eval >= num_groups
            infered_groups = sum((most_likely_cluster == (v(v ~= -1))') .* (1:num_groups), 2);
        else
            infered_groups = sum((most_likely_cluster == v') .* (1:num_groups), 2);
        end

        % evaluate goodness
        correct(i) = 1/size(PCAs,1) * sum((infered_groups - X(:,8) == 0));
        i = i + 1;
    end

    % best option
    [best_score_ori(cnt), best_idx] = max(correct);
    best_ori{cnt} = actual_permutes(best_idx,:);

    cnt = cnt + 1;
end
%%
% 
%     % -- cyl --
%     par_eval = GMMClassTrain(PCAs, num_gau_eval, seed);
%     gmPDF = @(x) pdf(par_eval.gm,x);
% 
%     [~, most_likely_cluster] = max(posterior(par_eval.gm, PCAs),[],2);
% 
% 
%     % try out all num_gau_plot!/(num_gau_plot - num_groups)! possible interpretations of the centres
%     num_groups = length(unique(cyls));
%     num_v = factorial(max(num_gau_eval, num_groups))/factorial(abs(num_gau_eval - num_groups));
% 
%     %permutes = perms([1:min(num_gau_plot, num_groups), zeros(1,abs(num_gau_plot - num_groups))]);
%     %actual_permutes = unique(permutes,'rows');
% 
%     permutes = perms(1:max(num_gau_eval, num_groups));
%     permutes(ismember(permutes, ( (min(num_groups,num_gau_eval)+1):max(num_gau_eval, num_groups) ) ) ) = -1;
%     actual_permutes = unique(permutes, 'rows');
% 
%     correct = zeros(num_v, 1);
%     i = 1;
%     for v = actual_permutes'
%         % assign groups
%         if num_gau_eval >= num_groups
%             infered_groups = sum( (most_likely_cluster == (v(v ~= -1))' ) .* (1:num_groups), 2);
%         else
%             infered_groups = sum( (most_likely_cluster == v')             .* (1:num_groups), 2);
%         end
% 
%         % evaluate goodness
%         correct(i) = 1/size(PCAs,1) * sum((infered_groups - sum((cyls == unique(cyls)') .* (1:num_groups), 2) == 0));
%         i = i + 1;
%     end
% 
%     % best option
%     [best_score_cyl(cnt), best_idx] = max(correct);
%     best_cyl{cnt} = actual_permutes(best_idx,:);
% 
% 
%     cnt = cnt + 1;
% 
% end
















