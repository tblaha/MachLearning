%% Hierarchical clustering

importdata_2018;
Y=X(:,2);
X=[X(:,1), X(:,3:end)];

Ynn=Xnn(:,2);
Xnn=[Xnn(:,1), Xnn(:,3:end)];

% Maximum number of clusters
Maxclust = 3;

% Compute hierarchical clustering
Z = linkage(X, 'ward', 'euclidean');

Znn = linkage(Xnn, 'ward', 'euclidean');

% Compute clustering by thresholding the dendrogram
i = cluster(Z, 'Maxclust', Maxclust);
inn = cluster(Znn, 'Maxclust', Maxclust);

%% Plot results

% Plot dendrogram
mfig('Dendrogram'); clf;
dendrogram(Z,0);

mfig('Dendrogram - not normalized'); clf;
dendrogram(Znn,0);

% Plot data
mfig('Hierarchical'); clf; 
clusterplot(X, Y, i);

mfig('Hierarchical - not normalized'); clf; 
clusterplot(Xnn, Ynn, inn);