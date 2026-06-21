function results = performGridSearchDS(u_train, y_train, u_val, y_val, u_test, y_test, M_vec, D_vec, R, MAXSWEEPS, runs, modeReg, modeRound, strategyInit, evaluateOptions, filenameTag)
% Sept. 2025 - Eva Memmel
% ------------------------------------------


% strategyInit.ALSinit = false;
runs_total = length(MAXSWEEPS);
strategyList = [repmat({'ALSinit'}, 1, strategyInit.ALSinit * runs_total)];

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
results.sweeps = cell(1, runs_total);


fprintf('Performing Grid Search over D and sweeps...\n');
for run = 1:runs_total
    D_max = D_vec(end);
    M = M_vec(1);
    sweeps =  MAXSWEEPS(run);
    counter = 1;


    strategy = strategyList{run};
    results.method{run} = strategy;
    results.M_value{run}.D{1} = M_vec(1);
    results.D_value{run}.D{1} = 2;
      

    % Initialization
    if strcmp(strategy, 'ALSinit')
        ranks = [1]; 
        fprintf('Grid Point Run %d/%d for D=%d, sweeps=%d...\n', run, runs_total, D_max, sweeps);
        if sweeps == 1
            MAXITR = computeMAXITR(2, sweeps+1) ;
        else
            MAXITR = computeMAXITR(2, sweeps) ;
        end
        N_train = length(u_train);
        train = tic();
        [TN_train, e, yhat_train] = mvals(u_train, y_train, M, [1], MAXITR, modeReg, 'linear',N_train);
        results.timetrain{run}.D{counter} = toc(train);

        % Store results
        D_curr = 2;
        results.etrain{run}.D{counter} = e;
        results.yhat_train{run}.D{counter} = yhat_train;
        results.sweeps{run} = sweeps;
        % store configuration

        if evaluateOptions.StoreTN
            results.TN_train{run}.D{counter} = TN_train;
        end

        if evaluateOptions.ValidationData
            val = tic();
            [results.val{run}.D{counter}, results.yhat_val{run}.D{counter}] = evaluateResults(TN_train, u_val, y_val, M);
            results.timeval{run}.D{counter} = toc(val);
        end

        if evaluateOptions.TestData    
            train_full = tic();
            [TN_full, e_full, yhat_train_full] = mvals([u_train; u_val], [y_train; y_val], M, [1], MAXITR, modeReg, 'linear',N_train);
            results.timetrain_full{run}.D{counter} = toc(train_full);
            test = tic();
            [results.test{run}.D{counter}, results.yhat_test{run}.D{counter}] = evaluateResults(TN_full, u_test, y_test, M);
            results.timetest{run}.D{counter} = toc(test);
            results.etrain_full{run}.D{counter} = e_full;
            results.TN_full{run}.D{counter} = TN_full;
        end

    else
        error('No valid initialization strategy selected.');
    end
    
    while D_curr < D_max
        counter = counter+1;
        increaseD_results = increaseD(TN_train, TN_full, u_train, y_train, u_val, y_val, u_test, y_test, M, R, [D_curr D_curr+1], sweeps, modeReg, evaluateOptions, filenameTag);
        fieldsToMerge = {'timetrain', 'D_value', 'etrain', 'yhat_train'};
        if evaluateOptions.ValidationData
            fieldsToMerge = [fieldsToMerge, {'val', 'yhat_val', 'timeval'}];
        end
        if evaluateOptions.TestData
            fieldsToMerge = [fieldsToMerge, {'test', 'yhat_test', 'timetest', 'etrain_full', 'timetrain_full'}];
        end
        if evaluateOptions.StoreTN
            fieldsToMerge = [fieldsToMerge, {'TN_train', 'TN_full'}];
        end
        for field = fieldsToMerge
            fieldName = field{1};
            newEntry = increaseD_results.(fieldName).D{1};
            results.(fieldName){run}.D{end+1} = newEntry;
        end
        D_curr = D_curr + 1;
        TN_train = results.TN_train{run}.D{counter};
        TN_full = results.TN_full{run}.D{counter};
        results.sweeps{run} = sweeps;
    end


end

end