help rfunction y_test_est = KNNExecute(par, X_test,features,outarg)

for arg = sort(outarg, 'desc')
        features(features > arg) = features(features > arg) - 1;
end

y_test_est=predict(par.knn, X_test(:,features));

end

