function results = increaseM(TN, u_train, y_train, u_val, y_val, u_test, y_test, M0, M_vec, R, D0, MAXSWEEPS, modeReg, evaluateOptions, filenameTag)
    results = struct();
    results.timetrain.M = cell(1, length(M_vec)-1);
    results.M_value.M = cell(1, length(M_vec)-1);
    results.etrain.M = cell(1, length(M_vec)-1);
    results.yhat_train.M = cell(1, length(M_vec)-1);
    results.val.M = cell(1, length(M_vec)-1);
    results.test.M = cell(1, length(M_vec)-1);
    results.yhat_val.M = cell(1, length(M_vec)-1);
    results.yhat_test.M = cell(1, length(M_vec)-1);
    results.timeval.M = cell(1, length(M_vec)-1);
    results.timetest.M = cell(1, length(M_vec)-1);
    results.TN_train.M = cell(1, length(M_vec)-1);

    fprintf('Running increaseM strategy...\n');

    ranks = R*ones(1,D0-1);
    for M_idx = 1:length(M_vec)-1
        M = M_vec(M_idx + 1);
        MAXITR = computeMAXITR(D0, MAXSWEEPS);
        fprintf('Increasing for M = %d\n', M);
        train = tic();
        [TN, e,yhat_train] = mvals(u_train,y_train,M,ranks,MAXITR,modeReg,'incM',TN);
        results.timetrain.M{M_idx} = toc(train);
        results.M_value.M{M_idx} = M;
        results.etrain.M{M_idx} = e;
        results.yhat_train.M{M_idx} = yhat_train;

        % If true, compute validation
        if evaluateOptions.ValidationData
            val = tic();
            [results.val.M{M_idx}, results.yhat_val.M{M_idx}] = evaluateResults(TN, u_val, y_val, M);
            results.timeval.M{M_idx} = toc(val);
        end
        if evaluateOptions.ValidationData
            test = tic();
            [results.test.M{M_idx}, results.yhat_test.M{M_idx}] = evaluateResults(TN, u_test, y_test, M);
            results.timetest.M{M_idx} = toc(test);
        end
        if evaluateOptions.StoreTN
            results.TN.M{M_idx} = TN; 
        end
    end
  %  save(sprintf('results/results_increaseD_%s.mat', filenameTag), 'results');
end
