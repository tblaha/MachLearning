function par = GMMClassTrain(PCAs, num_gau, seed)

    rng(seed)

    options = statset('Display','notify', 'MaxIter', 1000);
    par.gm = fitgmdist(PCAs, num_gau, 'Replicates', 10, 'Options', options);
    
end