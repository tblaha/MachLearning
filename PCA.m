clear
close all



%%
importdata_Report1;

%% linear algebra
[~,D_raw] = eig((X-mean(X))'*(X-mean(X))); % raw eigenvalues from the var-covar matrix
sigmas = sort(diag(D_raw), 'desc'); % sorted eigenvalues


[Ux,Dx,Px]  = svd( ( (X-mean(X)) ./ std(X) ) );
[Uy,Dy,Py]  = svd( ( (Y-mean(Y)) ./ std(Y) ) );
corrX       = corr((X-mean(X)) ./ std(X));

%% plot corrmap

corrmap = figure('Position', [0 0 800 600], 'visible', 'off');
imagesc(corrX);
cmap = hsv(256);
set(gca,'xticklabel',{'gpm','cylinders', 'displacement', 'horsepower', 'weight', 'accelleration', 'year', 'origin'})
xtickangle(60)
set(gca,'yticklabel',{'gpm','cylinders', 'displacement', 'horsepower', 'weight', 'accelleration', 'year', 'origin'})
xt = get(gca, 'XTick');
set(gca, 'FontSize', 16)
yt = get(gca, 'YTick');
set(gca, 'FontSize', 16)
colormap(fliplr(gray')')
colorbar
saveas(corrmap, strcat('Plots/corrmat.eps'),'epsc')

%%
fsize = 16;
fsizeldg = 14;



%% plots

%%%%%%
%%% TODO: all fonts bigger!
%%%%%%


% plot variance explaination plot
varplot = figure('Position',[0 0 800 600],'visible','off');
hold on
plot( 0:M, [0; cumsum(diag(Dx) / trace(Dx(1:8,1:8))) ], '-o')
plot( 0:M-1, [0; cumsum(diag(Dy) / trace(Dy(1:7,1:7))) ], '-x')
hold off
title('Variance Explaned','FontSize', fsize)
xlabel('Number of Principal Components','FontSize', fsize)
ylabel('Cumulative Normalized Variance','FontSize', fsize)
grid on
ylim([0,1])
lgd = legend('Cumulative \sigma of X', 'Cumulative \sigma of Y');
lgd.FontSize = fsizeldg;
lgd.Location = 'northwest';
saveas(varplot, strcat('Plots/varplot.eps'),'epsc')

% group by continent:
X1 = X(X(:, end) == 1, :);
X2 = X(X(:, end) == 2, :);
X3 = X(X(:, end) == 3, :);
Y1 = Y(Y(:, end) == 1, :);
Y2 = Y(Y(:, end) == 2, :);
Y3 = Y(Y(:, end) == 3, :);
Uy1 = Uy(Y(:, end) == 1, :);
Uy2 = Uy(Y(:, end) == 2, :);
Uy3 = Uy(Y(:, end) == 3, :);
Ux1 = Ux(X(:, end) == 1, :);
Ux2 = Ux(X(:, end) == 2, :);
Ux3 = Ux(X(:, end) == 3, :);

% plot 1st and 2nd principal components of corr(X)
prinplot12 = figure('Position',[0 0 800 600],'visible','off');
hold on
plot( Ux1*Dx(:,1), Ux1*Dx(:,2), 'o')
plot( Ux2*Dx(:,1), Ux2*Dx(:,2), 'x')
plot( Ux3*Dx(:,1), Ux3*Dx(:,2), '*')
hold off
grid on
title('Princial Component Analysis -- First 2 components','FontSize', fsize)
xlabel('X^T v_1 -- 1st principal component projections of X','FontSize', fsize)
ylabel('X^T v_2 -- 2st principal component projections of X','FontSize', fsize)
lgd = legend('American', 'European', 'Asian');
lgd.FontSize = fsizeldg;
lgd.Location = 'northwest';
saveas(prinplot12, strcat('Plots/1stPrin-2ndPrin.eps'),'epsc')

% plot 1st principal components of corr(Y) and output
% find regression of PCA1 vs output
A = [ones([Ny,1]), Uy*Dy(:,1) ];
b1 = A\X(:,1);
prinplot1out  = figure('Position',[0 0 800 600],'visible','off');
hold on
plot( (Uy1) * Dy(:,1), X1(:,1), 'o')
plot( (Uy2) * Dy(:,1), X2(:,1), 'x')
plot( (Uy3) * Dy(:,1), X3(:,1), '*')
x = linspace(-6, 6, 101);
plot( x, [ones([101 1]), x']*b1 )
hold off
grid on
title('Princial Component Analysis -- Output vs first component','FontSize', fsize)
xlabel('Y^T v_1 -- 1st principal component projections of Y','FontSize', fsize)
ylabel('Output [gallons per mile]','FontSize', fsize)
lgd = legend('American', 'European', 'Asian', 'linear regression');
lgd.FontSize = fsizeldg;
saveas(prinplot1out, strcat('Plots/1stPrin-Output.eps'),'epsc')

% plot 1st and 2nd principal component of corr(Y) and output. Save this one
% manually after selecting some nice viewing directions!
prinplot12out = figure('Position',[0 0 800 600],'visible','off');
hold on
plot3( (Y1-mean(Y1)) * Py(:,1), (Y1-mean(Y1)) * Py(:,2), X1(:,1), 'o')
plot3( (Y2-mean(Y2)) * Py(:,1), (Y2-mean(Y2)) * Py(:,2), X2(:,1), 'x')
plot3( (Y3-mean(Y3)) * Py(:,1), (Y3-mean(Y3)) * Py(:,2), X3(:,1), '*')
hold off
grid on
title('Princial Component Analysis -- Output vs first 2 components')
xlabel('Y^T v_1 -- 1st principal component projections of Y')
ylabel('Y^T v_2 -- 2st principal component projections of Y')
zlabel('Output [gallons per mile]')
legend('American', 'European', 'Asian')

% plot component pattern PCA1 vs PCA2
comppat = figure('Position', [0 0 600 600], 'visible', 'off');
t = linspace(0, 2*pi, 101);
hold on
    plot(sin(t), cos(t))
    i = 1;
    for a = [Px(:,1)' ; Px(:,2)']
        quiver(0, 0, a(1), a(2), 'k','linewidth',1.5);
        if i == 3
            dispa = a*1.3 + [-0.1 0.1]';
        elseif i == 6
            dispa = a*1.3 + [0.15 0]';
        elseif i == 7
            dispa = a*1.1;
        else
            dispa = a*1.4;
        end
        text(dispa(1),dispa(2), data.Properties.VariableNames{2+i},...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle','FontSize',14)
        i = i+1;
    end
hold off
grid on
xlim([-1.2,1.2])
ylim([-1.2,1.2])
axis equal
title('Component pattern PCA1,2','FontSize', fsize)
xlabel('1st principal component projections of X','FontSize', fsize)
ylabel('2nd principal component projections of X','FontSize', fsize)
saveas(comppat, strcat('Plots/ComPatPCA1PCA2.eps'),'epsc')


% plot component pattern PCA1 vs output
comppatout = figure('Position', [0 0 800 600], 'visible', 'off');
hold on
B = b1' * [zeros([7, 1]), Py(:,1)]';
bar(B);
set(gca,'xticklabel',{'','cylinders', 'displacement', 'horsepower', 'weight', 'accelleration', 'year', 'origin'})
xtickangle(60)
xt = get(gca, 'XTick');
set(gca, 'FontSize', 16)
hold off
grid on
title('Component pattern PCA1/Output','FontSize', fsize)
xlabel('1st principal component','FontSize', fsize)
ylabel('Impact on fuel comsumption (gallons per mile)','FontSize', fsize)
saveas(comppatout, strcat('Plots/ComPatPCA1Output.eps'),'epsc')


%% evaluating the regression thing

beta = zeros(2,7);
for i = 1:7
    A = [ones([Ny,1]), Uy*Dy(:,i) ];
    beta(:,i) = A\X(:,1);
end

B = beta(2,:).*Py(:,:); % yields 7x7 matrix, multiplying the linear regression coeffients to the corresponsing eigenvectors


inferredgpm = zeros([length(X), 7]);

for c = 1:7
    
    % c components
    for i = 1:length(X)
        inferredgpm(i,c) = 0;
        for j = 1:c
            inferredgpm(i,c) = inferredgpm(i,c) + (B(j,:)*((Y(i,1:end)-mean(Y))./std(Y))'); % subtracting the infered value from the actual value and squaring
        end
    end
    
    inferredgpm(:,c) = beta(1,1) + inferredgpm(:,c); % adding back in the constant regression term
    
end

% one components
meanoffsetone = mean(inferredgpm(:,1)) - mean(X(:,1)); % should be zero by definition
varianceone = sum((inferredgpm(:,1) - X(:,1)).^2 ./ length(X));
sdone       = sqrt(varianceone);
disp(sdone/mean(X(:,1)))

% all components
meanoffset = mean(inferredgpm(:,7)) - mean(X(:,1)); % should be zero by definition
variance = sum((inferredgpm(:,7) - X(:,1)).^2 ./ length(X));
sd       = sqrt(variance);
disp(sd/mean(X(:,1)))

%plotting the above
regressout = figure('Position', [0 0 800 500], 'visible', 'on');
hold on
boxplot(sqrt((inferredgpm - X(:,1)).^2 ./ length(X) )) 
%set(gca,'xticklabel',{'','cylinders', 'displacement', 'horsepower', 'weight', 'accelleration', 'year', 'origin'})
%xtickangle(60)
xt = get(gca, 'XTick');
set(gca, 'FontSize', 16)
hold off
grid on
title('Terms of the standard error of the regressions','FontSize', fsize)
xlabel('Amount of Principal Components included','FontSize', fsize)
ylabel('Fuel comsumption (gallons per mile)','FontSize', fsize)
saveas(regressout, strcat('Plots/regressout.eps'),'epsc')

