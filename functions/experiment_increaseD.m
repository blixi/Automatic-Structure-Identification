function results = experiment_increaseD(u_train, y_train, u_val, y_val, u_test, y_test, M0, R, D0, D_vec, MAXSWEEPS, totalMAXITR, runs, modeReg, modeRound, strategyInit, evaluateOptions, filenameTag)
% Experiment function for incremental D strategy

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
results.TN_train = cell(1, runs_total);
results.D_value = cell(1, runs_total);

fprintf('Running IncreaseD Experiment...\n');

for run = 1:runs_total
    strategy = strategyList{run};
    fprintf('Run %d/%d using %s\n', run, runs_total, strategy);
    if strcmp(strategy, 'SOTA')
        % SOTA runs: Train models from scratch for each D in D_vec
        results.method{run} = 'SOTA';
        for D_idx = 1:length(D_vec)
            D = D_vec(D_idx);
            ranks = R * ones(1, D - 1);

            fprintf('SOTA MVALS Run %d/%d for D=%d\n', run, runs_total, D);
            train = tic();
            [TN_train, e, yhat_train] = mvals(u_train, y_train, M0, ranks, totalMAXITR, modeReg, 'inc0');
            results.timetrain{run}.D{D_idx} = toc(train);
            results.D_value{run}.D{D_idx} = D;
            results.etrain{run}.D{D_idx} = e;
            results.yhat_train{run}.D{D_idx} = yhat_train;

            if evaluateOptions.ValidationData
                val = tic();
                [results.val{run}.D{D_idx}, results.yhat_val{run}.D{D_idx}] = evaluateResults(TN_train, u_val, y_val, M0);
                results.timeval{run}.D{D_idx} = toc(val);
            end
            if evaluateOptions.TestData
                test = tic();
                [results.test{run}.D{D_idx}, results.yhat_test{run}.D{D_idx}] = evaluateResults(TN_train, u_test, y_test, M0);
                results.timetest{run}.D{D_idx} = toc(test);
            end
            if evaluateOptions.StoreTN
                results.TN_train{run}.D{D_idx} = TN_train;
            end
        end
    else
        MAXITR = computeMAXITR(D0,MAXSWEEPS)+1; %initialALS starts at 1;
        ranks = R*ones(1,D0-1);
        % Initialize TN based on chosen strategy
        if strcmp(strategy, 'ALSinit')
            results.method{run} = 'ALSinit';
            fprintf('Running MVALS (ALSinit) for initial training...\n');
            train = tic();
            [TNinit, e, yhat_train] = mvals(u_train, y_train, M0, ranks, MAXITR, modeReg, 'inc0');
            results.timetrain{run}.D{1} = toc(train);
            [TNinit_full, e, yhat_train] = mvals([u_train; u_val], [y_train; y_val], M0, ranks, MAXITR, modeReg, 'inc0');
        elseif strcmp(strategy, 'SVDinit')
            results.method{run} = 'SVDinit';
            fprintf('Running MVSVD (SVDinit) for initial training...\n');
            train = tic();
            [TNinit, e, yhat_train] = mvsvd(u_train, y_train, M0, D0, modeRound);
            results.timetrain{run}.D{1} = toc(train);
            [TNinit_full, e, yhat_train] = mvsvd([u_train; u_val], [y_train; y_val], M0, D0, modeRound);
        end
        results.D_value{run}.D{1} = D0;
        results.etrain{run}.D{1} = e;
        results.yhat_train{run}.D{1} = yhat_train;

        if evaluateOptions.ValidationData
            val = tic();
            [results.val{run}.D{1}, results.yhat_val{run}.D{1}] = evaluateResults(TNinit, u_val, y_val, M0);
            results.timeval{run}.D{1} = toc(val);
        end
        if evaluateOptions.TestData
            test = tic();
            [results.test{run}.D{1}, results.yhat_test{run}.D{1}] = evaluateResults(TNinit_full, u_test, y_test, M0);
            results.timetest{run}.D{1} = toc(test);
        end
        if evaluateOptions.StoreTN
            results.TN_train{run}.D{1} = TNinit;
        end

        % Call increaseD for the selected initial TN
        fprintf('Running increaseD ...\n');
        increaseD_results = increaseD(TNinit, TNinit_full, u_train, y_train, u_val, y_val, u_test, y_test, M0, R, D_vec, MAXSWEEPS, modeReg, evaluateOptions, filenameTag);
        fieldsToMerge = {'timetrain', 'D_value', 'etrain', 'yhat_train'};
        if evaluateOptions.ValidationData
            fieldsToMerge = [fieldsToMerge, {'val', 'yhat_val', 'timeval'}];
        end
        if evaluateOptions.TestData
            fieldsToMerge = [fieldsToMerge, {'test', 'yhat_test', 'timetest'}];
        end
        if evaluateOptions.StoreTN
            fieldsToMerge = [fieldsToMerge, {'TN_train'}];
        end
        for field = fieldsToMerge
            fieldName = field{1};
            results.(fieldName){run}.D = [results.(fieldName){run}.D, increaseD_results.(fieldName).D];
        end
    end

end

% Save results
save(sprintf('results/experiment_increaseD/results_increaseD_%s.mat', filenameTag), 'results');
end
