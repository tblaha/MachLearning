%% Hierarchical clustering


datann=load('Xnn.mat'); % not normalizes values
Xnn=datann.Xnn;



Ynn=Xnn(:,2);
Xnn=[Xnn(:,1), Xnn(:,3:end)];

% Maximum number of clusters
Maxclust = 3;

% Compute hierarchical clustering


Znn = linkage(Xnn, 'ward', 'euclidean');

% Compute clustering by thresholding the dendrogram

inn = cluster(Znn, 'Maxclust', Maxclust);



%% Plot results

% Plot dendrogram

mfig('Dendrogram - not normalized'); clf;
dendrogram(Znn,30);

% Plot data

mfig('Hierarchical - not normalized'); clf; 
clusterplot(Xnn, Ynn, inn);