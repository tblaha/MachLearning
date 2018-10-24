clear
close all

%%
importdata_Report1

%% 
mean_x   = mean(X);
std_x    = std(X);
median_x = median(X);
range_x  = range(X);

%% Histogram
fsize = 20;
Year = data{:,9};
figure('Position',[0 0 800 600],'visible','off');
histplot=histogram(Year);
title('Car model year','FontSize', fsize)
xlabel('Year manufactured','FontSize', fsize)
ylabel('Amount of cars','FontSize', fsize)
ylim([0,45]);
xlim([69,83]);
saveas(histplot, strcat('Plots/histplot.eps'),'epsc')

%% Histogram
fsize = 20;
HP = data{:,6};
figure('Position',[0 0 800 600],'visible','off');
histplot2=histogram(HP);
title('Car horsepower','FontSize', fsize)
xlabel('Horsepower','FontSize', fsize)
ylabel('Amount of cars','FontSize', fsize)
% ylim([0,45]);
% xlim([69,83]);
saveas(histplot2, strcat('Plots/histplot2.eps'),'epsc')


%% by continent

X1 = X(X(:, end) == 1, :);
X2 = X(X(:, end) == 2, :);
X3 = X(X(:, end) == 3, :);

%% scatter plots
sc1 = figure('Position',[0 0 800 600],'visible','off');
hold on
plot( X1(:,3), X1(:,4), 'o')
plot( X2(:,3), X2(:,4), 'x')
plot( X3(:,3), X3(:,4), '*')
hold off
grid on
title('Scatterplot of displacement and power','FontSize', fsize)
xlabel('Displacement [cubic inch]','FontSize', fsize)
ylabel('Horsepower [cubic inch]','FontSize', fsize)
lgd = legend('American', 'European', 'Asian');
lgd.FontSize = 14;
%lgd.Location = 'northwest';
saveas(sc1, strcat('Plots/scatter4vs3.eps'),'epsc')


sc2 = figure('Position',[0 0 800 600],'visible','off');
hold on
plot( X1(:,7), X1(:,5), 'o')
plot( X2(:,7), X2(:,5), 'x')
plot( X3(:,7), X3(:,5), '*')
hold off
grid on
title('Scatterplot of year and weight','FontSize', fsize)
ylabel('Weight [lbs]','FontSize', fsize)
xlabel('Year','FontSize', fsize)
lgd = legend('American', 'European', 'Asian');
lgd.FontSize = 14;
%lgd.Location = 'northwest';
saveas(sc2, strcat('Plots/scatter7vs5.eps'),'epsc')
