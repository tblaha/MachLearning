function par = KNNTrain(X_train,output,neighbours)

Distance = 'euclidean'; % Distance measure
par.knn=fitcknn(X_train, output, 'NumNeighbors', neighbours, 'Distance', Distance);

end

