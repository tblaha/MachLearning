clear
close all

%%
importdata

%% 
mean_x   = mean(X);
std_x    = std(X);
median_x = median(X);
range_x  = range(X);

%% Histogram
Year = data{:,9};
histplot=histogram(Year);
title('Car model year')
xlabel('Year manufactured')
ylabel('Amount of cars')
ylim([0,45]);
xlim([69,83]);
saveas(histplot, strcat('Plots/histplot.eps'),'epsc')