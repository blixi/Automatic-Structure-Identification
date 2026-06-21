function results = increaseD(TN_train, TN_full, u_train, y_train, u_val, y_val, u_test, y_test, M0, R, D_vec, MAXSWEEPS, modeReg, evaluateOptions, filenameTag)
    results = struct();
    results.timetrain.D = cell(1, length(D_vec)-1);
    results.timetrain_full.D = cell(1, length(D_vec)-1);
    results.D_value.D = cell(1, length(D_vec)-1);
    results.etrain.D = cell(1, length(D_vec)-1);
    results.etrain_full.D = cell(1, length(D_vec)-1);
    results.yhat_train.D = cell(1, length(D_vec)-1);
    results.val.D = cell(1, length(D_vec)-1);
    results.test.D = cell(1, length(D_vec)-1);
    results.yhat_val.D = cell(1, length(D_vec)-1);
    results.yhat_test.D = cell(1, length(D_vec)-1);
    results.timeval.D = cell(1, length(D_vec)-1);
    results.timetest.D = cell(1, length(D_vec)-1);
    results.TN_train.D = cell(1, length(D_vec)-1);
    results.TN_full.D = cell(1, length(D_vec)-1);

    fprintf('Running increaseD strategy...\n');
    for D_idx = 1:length(D_vec)-1
        
        D = D_vec(D_idx + 1);
        MAXITR = computeMAXITR(D, MAXSWEEPS);
        ranks = R*ones(1,D-1);
        fprintf('Increasing for D = %d\n', D);
        train = tic();
        [TN_train, e,yhat_train] = mvals(u_train,y_train,M0,ranks,MAXITR,modeReg,'incD',TN_train);
        results.timetrain.D{D_idx} = toc(train);
        results.D_value.D{D_idx} = D;
        results.etrain.D{D_idx} = e;
        results.yhat_train.D{D_idx} = yhat_train;
    
        % If true, compute validation
        if evaluateOptions.ValidationData
            val = tic();
            [results.val.D{D_idx}, results.yhat_val.D{D_idx}] = evaluateResults(TN_train, u_val, y_val, M0);
            results.timeval.D{D_idx} = toc(val);
        end
        if evaluateOptions.TestData
            train_full = tic();
            [TN_full, e_full,yhat_train_full] = mvals([u_train; u_val],[y_train; y_val],M0,ranks,MAXITR,modeReg,'incD',TN_full);
            results.timetrain_full.D{D_idx} = toc(train_full);
            test = tic();
            [results.test.D{D_idx}, results.yhat_test.D{D_idx}] = evaluateResults(TN_full, u_test, y_test, M0);
            results.timetest.D{D_idx} = toc(test);
            results.etrain_full.D{D_idx} = e_full;
        end
        if evaluateOptions.StoreTN
            results.TN_train.D{D_idx} = TN_train; 
            results.TN_full.D{D_idx} = TN_full;
        end
    end
  %  save(sprintf('results/results_increaseD_%s.mat', filenameTag), 'results');
end
