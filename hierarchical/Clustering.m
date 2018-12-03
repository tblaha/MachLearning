%% Hierarchical clustering

data=load('X.mat');
X=data.X;

datann=load('Xnn.mat'); % not normalizes values
Xnn=datann.Xnn;


Y=Xnn(:,2);

X=[X(:,1), X(:,3:end)];


% Maximum number of clusters
Maxclust = 3;

% Compute hierarchical clustering
Z = linkage(X, 'ward', 'euclidean');


% Compute clustering by thresholding the dendrogram
i = cluster(Z, 'Maxclust', Maxclust);


%% Plot results

% Plot dendrogram
mfig('Dendrogram'); clf;
dendrogram(Z,30);

% Plot data
mfig('Hierarchical'); clf; 
clusterplot(X, Y, i);





