function results = experiment_increaseM(u_train, y_train, u_val, y_val, u_test, y_test, M0, M_vec, R, D0, MAXSWEEPS, totalMAXITR, runs, modeReg, modeRound, strategyInit, evaluateOptions, filenameTag)
% Experiment function for incremental M strategy
% Sept. 2025 - Eva Memmel
% ------------------------------------------

runs_total = strategyInit.ALSinit * runs.runsALS + strategyInit.SVDinit * runs.runsSVD + strategyInit.SOTA * runs.runsALS;
strategyList = [repmat({'ALSinit'}, 1, runs.runsALS), repmat({'SVDinit'}, 1, runs.runsSVD), repmat({'SOTA'}, 1, runs.runsALS)];

results = struct();
results.method = cell(1, runs_total);
results.timetrain = cell(1, runs_total);
results.timeval = cell(1, runs_total);
results.timetest = cell(1, runs_total);
results.etrain = cell(1, runs_total);
results.val = cell(1, runs_total);
results.test = cell(1, runs_total);
results.yhat_train = cell(1, runs_total);
results.yhat_val = cell(1, runs_total);
results.TN = cell(1, runs_total);
results.M_value = cell(1, runs_total);

fprintf('Running IncreaseM Experiment...\n');

ranks = R * ones(1, D0 - 1);
for run = 1:runs_total
    strategy = strategyList{run};
    fprintf('Run %d/%d using %s\n', run, runs_total, strategy);
    if strcmp(strategy, 'SOTA')
        % SOTA runs: Train models from scratch for each D in D_vec
        results.method{run} = 'SOTA';
        for M_idx = 1:length(M_vec)
            M = M_vec(M_idx);
            fprintf('SOTA MVALS Run %d/%d for M=%d\n', run, runs_total, M);
            train = tic();
            [TN, e, yhat_train] = mvals(u_train, y_train, M, ranks, totalMAXITR, modeReg, 'inc0');
            results.timetrain{run}.M{M_idx} = toc(train);
            results.M_value{run}.M{M_idx} = M;
            results.etrain{run}.M{M_idx} = e;
            results.yhat_train{run}.M{M_idx} = yhat_train;

            if evaluateOptions.ValidationData
                val = tic();
                [results.val{run}.M{M_idx}, results.yhat_val{run}.M{M_idx}] = evaluateResults(TN, u_val, y_val, M);
                results.timeval{run}.M{M_idx} = toc(val);
            end
            if evaluateOptions.TestData
                test = tic();
                [results.test{run}.M{M_idx}, results.yhat_test{run}.M{M_idx}] = evaluateResults(TN, u_test, y_test, M);
                results.timetest{run}.M{M_idx} = toc(test);
            end
            if evaluateOptions.StoreTN
                results.TN{run}.M{M_idx} = TN;
            end
        end
    else
        MAXITR = computeMAXITR(D0,MAXSWEEPS) + 1; %intial ALS starts at 1
        % Initialize TN based on chosen strategy
        if strcmp(strategy, 'ALSinit')
            results.method{run} = 'ALSinit';
            fprintf('Running MVALS (ALSinit) for initial training...\n');
            train = tic();
            [TNinit, e, yhat_train] = mvals(u_train, y_train, M0, ranks, MAXITR, modeReg, 'inc0');
            results.timetrain{run}.M{1} = toc(train);
        elseif strcmp(strategy, 'SVDinit')
            results.method{run} = 'SVDinit';
            fprintf('Running MVSVD (SVDinit) for initial training...\n');
            train = tic();
            [TNinit, e, yhat_train] = mvsvd(u_train, y_train, M0, D0, modeRound);
            results.timetrain{run}.M{1} = toc(train);
        end
        results.M_value{run}.M{1} = M0;
        results.etrain{run}.M{1} = e;
        results.yhat_train{run}.M{1} = yhat_train;

        if evaluateOptions.ValidationData
            val = tic();
            [results.val{run}.M{1}, results.yhat_val{run}.M{1}] = evaluateResults(TNinit, u_val, y_val, M0);
            results.timeval{run}.M{1} = toc(val);
        end
        if evaluateOptions.TestData
            test = tic();
            [results.test{run}.M{1}, results.yhat_test{run}.M{1}] = evaluateResults(TNinit, u_test, y_test, M0);
            results.timetest{run}.M{1} = toc(test);
        end
        if evaluateOptions.StoreTN
            results.TN{run}.M{1} = TNinit;
        end

        % Call increaseM for the selected initial TN
        fprintf('Running increaseM ...\n');
        increaseM_results = increaseM(TNinit, u_train, y_train, u_val, y_val, u_test, y_test, M0, M_vec, R, D0, MAXSWEEPS, modeReg, evaluateOptions, filenameTag);
        fieldsToMerge = {'timetrain', 'M_value', 'etrain', 'yhat_train'};
        if evaluateOptions.ValidationData
            fieldsToMerge = [fieldsToMerge, {'val', 'yhat_val', 'timeval'}];
        end
        if evaluateOptions.TestData
            fieldsToMerge = [fieldsToMerge, {'test', 'yhat_test', 'timetest'}];
        end
        if evaluateOptions.StoreTN
            fieldsToMerge = [fieldsToMerge, {'TN'}];
        end
        for field = fieldsToMerge
            fieldName = field{1};
            results.(fieldName){run}.M = [results.(fieldName){run}.M, increaseM_results.(fieldName).M];
        end
    end

end
    % Save results
    save(sprintf('results/experiment_increaseM/results_increaseM_%s.mat', filenameTag), 'results');
end
