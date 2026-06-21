function results = experiment_increase0(u_train, y_train, u_val, y_val, u_test, y_test, M0, R, D0, MAXITR, runs, modeReg, modeRound, strategyInit, evaluateOptions, filenameTag)
% Sept. 2025 - Eva Memmel
% ------------------------------------------
results = struct(); 
results.method = cell(1, runs.runsALS + runs.runsSVD);
results.timetrain = cell(1, runs.runsALS + runs.runsSVD);
results.timeval = cell(1, runs.runsALS + runs.runsSVD);
results.timetest = cell(1, runs.runsALS + runs.runsSVD);
results.etrain = cell(1, runs.runsALS + runs.runsSVD);
results.val = cell(1, runs.runsALS + runs.runsSVD);
results.test = cell(1, runs.runsALS + runs.runsSVD);
results.yhat_train = cell(1, runs.runsALS + runs.runsSVD);
results.yhat_val = cell(1, runs.runsALS + runs.runsSVD);
results.TN = cell(1, runs.runsALS + runs.runsSVD);
results.D_value = cell(1, runs.runsALS + runs.runsSVD);

fprintf('Running Increase0 Experiment...\n');

% runs_total = strategyInit.ALSinit*runs.runsALS + strategyInit.SVDinit*runs.runsSVD;
runs_total = strategyInit.ALSinit*runs.runsALS;
% strategyList = [repmat({'ALSinit'}, 1, runs.runsALS), repmat({'SVDinit'}, 1, runs.runsSVD)];
strategyList = [repmat({'ALSinit'}, 1, runs.runsALS)];
ranks = R*ones(1,D0-1);

% Run standard MVALS (inc0)
for run = 1:runs_total
    strategy = strategyList{run};
    fprintf('Run %d/%d using %s\n', run, runs_total, strategy);
    % Run mvals
    if strcmp(strategy, 'ALSinit')
        results.method{run} = 'ALSinit';
        fprintf('Standard MVALS Run %d/%d\n', run, runs_total);
        train = tic();
        [TN, e, yhat_train] = mvals(u_train,y_train,M0,ranks,MAXITR,modeReg,'inc0');
        results.timetrain{run} = toc(train);
        % run mvsvd
    elseif strcmp(strategy, 'SVDinit')
        results.method{run} = 'SVDinit';
        fprintf('Standard MVSVD Run %d/%d\n', run, runs_total);
        train = tic();
        [TN,e,yhat_train] = mvsvd(u_train,y_train,M0,D0,modeRound);
        results.timetrain{run} = toc(train);
    end
    results.etrain{run} = e;
    results.yhat_train{run} = yhat_train;
    results.D_value{run} = D0;

    % If true, compute Validation, store TN
    if evaluateOptions.ValidationData
        val = tic();
        [results.val{run}, results.yhat_val{run}] = evaluateResults(TN, u_val, y_val, M0);
        results.timeval{run} = toc(val);
    end
    if evaluateOptions.TestData
        test = tic();
        [results.test{run}, results.yhat_test{run}] = evaluateResults(TN, u_test, y_test, M0);
        results.timetest{run} = toc(test);
    end
    if evaluateOptions.StoreTN
        results.TN{run} = TN;
    end
end

% Save results
save(sprintf('results/results_increase0_%s.mat', filenameTag), 'results');
end