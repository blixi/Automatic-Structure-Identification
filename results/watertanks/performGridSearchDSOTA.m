
function results = performGridSearchDSOTA(u_train, y_train, u_val, y_val, u_test, y_test, M_vec, D_vec, R, MAXITR, runs, modeReg, modeRound, strategyInit, evaluateOptions, filenameTag)
% Sept. 2025 - Eva Memmel
% ------------------------------------------

runs_total = strategyInit.ALSinit * runs.runsALS + strategyInit.SVDinit * runs.runsSVD + strategyInit.SOTA * runs.runsALS;
strategyList = [repmat({'ALSinit'}, 1, strategyInit.ALSinit *runs.runsALS), repmat({'SVDinit'}, 1, strategyInit.SVDinit *runs.runsSVD), repmat({'SOTA'}, 1, strategyInit.SOTA *runs.runsALS)];

% Initialize
results = struct();
results.method = cell(1, runs_total);
results.timetrain = cell(1, runs_total);
results.timetrain_full = cell(1, runs_total);
results.timeval = cell(1, runs_total);
results.timetest = cell(1, runs_total);
results.etrain = cell(1, runs_total);
results.etrain_full = cell(1, runs_total);
results.val = cell(1, runs_total);
results.test = cell(1, runs_total);
results.yhat_train = cell(1, runs_total);
results.yhat_val = cell(1, runs_total);
results.yhat_test = cell(1, runs_total);
results.TN_train = cell(1, runs_total);
results.TN_full = cell(1, runs_total);
results.D_value = cell(1, runs_total);
results.M_value = cell(1, runs_total);
results.seed = cell(1, runs_total);

fprintf('Running Watertanks Experiment SOTA VTN...\n');

for run = 1:runs_total
    counter = 1;
    for D_idx = 1:length(D_vec)
        D = D_vec(D_idx);
        M = M_vec(1);
        ranks = R * ones(1, D - 1);
        results.method{run} = 'SOTA';
        fprintf('SOTA MVALS Run %d/%d for D=%d, M=%d\n', run, runs_total, D, M);
        train = tic();
        [TN, e, yhat_train,~,rndseed] = mvals(u_train, y_train, M, ranks, MAXITR, modeReg, 'inc0');
        results.timetrain{run}.D{counter} = toc(train);
        results.D_value{run}.D{counter} = D;
        results.M_value{run}.D{counter} = M;
        results.etrain{run}.D{counter} = e;
        results.yhat_train{run}.D{counter} = yhat_train;
        results.seed{run}.D{counter} = rndseed;
        if evaluateOptions.ValidationData
            test = tic();
            [results.val{run}.D{counter}, results.yhat_val{run}.D{counter}] = evaluateResults(TN, u_val, y_val, M);
            results.timeval{run}.D{counter} = toc(test);
        end
        if evaluateOptions.StoreTN
            results.TN{run}.D{counter} = TN;
        end

        if evaluateOptions.ValidationData
            test = tic();
            [results.val{run}.D{counter}, results.yhat_val{run}.D{counter}] = evaluateResults(TN, u_val, y_val, M);
            results.timeval{run}.D{counter} = toc(test);
        end

        if evaluateOptions.StoreTN
            results.TN{run}.D{counter} = TN;
        end

        if evaluateOptions.TestData
            train_full = tic();
            [TN_full, e_full, yhat_train_full,~, rndseed_full] = mvals([u_train; u_val], [y_train; y_val], M, ranks, MAXITR, modeReg, 'inc0seed',rndseed);
            if rndseed_full ~= rndseed
                warning('random seed for validation and test data is not identical')
            end
            results.timetrain_full{run}.D{counter} = toc(train_full);
            test = tic();
            [results.test{run}.D{counter}, results.yhat_test{run}.D{counter}] = evaluateResults(TN_full, u_test, y_test, M);
            results.timetest{run}.D{counter} = toc(test);
            results.etrain_full{run}.D{counter} = e_full;
            results.TN_full{run}.D{counter} = TN_full;
        end
        counter = counter + 1;
    end




end

end