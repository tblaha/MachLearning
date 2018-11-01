function par = KNNTrain(X_train,features,outarg,neighbours)

output=X_train(:,outarg);
Distance = 'euclidean'; % Distance measure
par.knn=fitcknn(X_train(:,features), output, 'NumNeighbors', neighbours, 'Distance', Distance);

end

