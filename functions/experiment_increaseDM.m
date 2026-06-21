function results = experiment_increaseDM(u_train, y_train, u_val, y_val, u_test, y_test, M_vec, D_vec, R, MAXSWEEPS, runs, modeReg, modeRound, strategyInit, evaluateOptions, filenameTag)
% Sept. 2025 - Eva Memmel
% ------------------------------------------

runs_total = strategyInit.ALSinit * runs.runsALS + strategyInit.SVDinit * runs.runsSVD + strategyInit.SOTA * runs.runsALS;
strategyList = [repmat({'ALSinit'}, 1, runs.runsALS), repmat({'SVDinit'}, 1, runs.runsSVD), repmat({'SOTA'}, 1, runs.runsALS)];


% Initialize
results = struct();
results.method = cell(1, runs_total);
results.timetrain = cell(1, runs_total);
results.timeval = cell(1, runs_total);
results.timetest = cell(1, runs_total);
results.y_and_res_val = cell(1,runs_total);
results.etrain = cell(1, runs_total);
results.val = cell(1, runs_total);
results.test = cell(1, runs_total);
results.yhat_train = cell(1, runs_total);
results.yhat_val = cell(1, runs_total);
results.TN_train = cell(1, runs_total);
results.D_value = cell(1, runs_total);
results.M_value = cell(1, runs_total);
results.R_value = cell(1, runs_total);
results.updated = cell(1, runs_total);

R0 = R;
R_kick = 1;

fprintf('Running IncreaseDM Experiment...\n');
for run = 1:runs_total
    strategy = strategyList{run};
    fprintf('Run %d/%d using %s\n', run, runs_total, strategy);
    if strcmp(strategy, 'SOTA')
        disp('SOTA disabled for now')
    else
        % Start values
        M_idx = 1;
        D_idx = 1;
        M = M_vec(M_idx);
        D = D_vec(D_idx);
        ranks = R * ones(1, D - 1);
        R = R0;

        % --- INITIALIZATION ---
        if strcmp(strategy, 'SVDinit')
            results.method{run} = 'SVDinit';
            fprintf('Running MVSVD (SVDinit) for initial training with M_init=%d, D_init=%d...\n', M, D);
            train = tic();
            [TN, e, yhat_train] = mvsvd(u_train, y_train, M, D, modeRound);
            results.timetrain{run}.DM{1} = toc(train);
        elseif strcmp(strategy, 'ALSinit')
            results.method{run} = 'ALSinit';
            fprintf('Running MVALS (ALSinit) for initial training with M_init=%d, D_init=%d...\n', M, D);
            MAXITR = computeMAXITR(D, MAXSWEEPS) + 1;
            train = tic();
            [TN, e, yhat_train] = mvals(u_train, y_train, M, ranks, MAXITR, modeReg, 'inc0');
            results.timetrain{run}.DM{1} = toc(train);
        else
            error('No valid initialization strategy selected.');
        end
        results.D_value{run}.DM{1} = D;
        results.M_value{run}.DM{1} = M;
        results.R_value{run}.DM{1} = R;
        results.etrain{run}.DM{1} = e;
        results.y_and_res_val{run}.DM{1} = 1;
        results.yhat_train{run}.DM{1} = yhat_train;

        res = yhat_train-y_train;

        if evaluateOptions.ValidationData
            val = tic();
            [results.val{run}.DM{1}, results.yhat_val{run}.DM{1}] = evaluateResults(TN, u_val, y_val, M);
            results.timeval{run}.DM{1} = toc(val);
        end
        if evaluateOptions.TestData
            test = tic();
            [results.test{run}.DM{1}, results.yhat_test{run}.DM{1}] = evaluateResults(TN, u_test, y_test, M);
            results.timetest{run}.DM{1} = toc(test);
        end
        if evaluateOptions.StoreTN
            results.TN_train{run}.DM{1} = TN;
        end

        relValError = results.val{run}.DM{1}.relError;
        previous_relValError = relValError;
        max_increase_count = 3;
        increase_count = 0;

        results.updated{run}.DM{1} = 'init';
        counter = 1;

        yhat_train_old = yhat_train;

        % --- ITERATIVE SEARCH ---
        while (M_idx < length(M_vec) || D_idx < length(D_vec)) &&...
                (increase_count <= max_increase_count) && ...
                (previous_relValError > 1e-5) && ...
                counter < 20

            canIncrease_D = D_idx < length(D_vec);
            canIncrease_M = M_idx < length(M_vec);
            % canIncrease_R = R < 4;

            
            % Try D update
            if canIncrease_D
                D_trial = D_vec(D_idx + 1);
                temp.D = increaseD(TN, TN, u_train, y_train, u_val, y_val, u_test, y_test, M, R, [D D_trial], MAXSWEEPS, modeReg, evaluateOptions, filenameTag);
                RMSE_D = temp.D.val.D{1}.RMSE;
                yhat_train_D =  temp.D.yhat_train.D{1};
                VAF_D = vaf(y_train(M+1:end),yhat_train_D(M+1:end)-yhat_train_old(M+1:end));
            end

            % Try M update
            if canIncrease_M
                M_trial = M_vec(M_idx + 1);
                temp.M = increaseM(TN, u_train, y_train, u_val, y_val, u_test, y_test, M, [M M_trial], [R], D, MAXSWEEPS, modeReg, evaluateOptions, filenameTag);
                RMSE_M = temp.M.val.M{1}.RMSE;
                yhat_train_M =  temp.M.yhat_train.M{1};
                VAF_M = vaf(y_train(M+1:end), yhat_train_M(M+1:end)-yhat_train_old(M+1:end));
            end

            % if canIncrease_R                
            %     temp.R = increaseR(TN, u_train, y_train, res, u_val, y_val, u_test, y_test, M, R_kick, D, MAXSWEEPS, modeReg, evaluateOptions);
            %     relValErr_R = temp.R.val.R{1}.relError;
            % end

            % Determine Update direction based on relative Error
            chosenDirection = chooseDirection(canIncrease_D, canIncrease_M, VAF_D, VAF_M);
            results.updated{run}.DM{end+1} = chosenDirection;

            if chosenDirection == 'D' || chosenDirection == 'M' || chosenDirection == 'R'
                dir = chosenDirection
            else
                error("No direction chosen. Something went wrong")
            end

            % Merge results of chosen updates to results.(fieldName).DM
            fieldsToMerge = {}; 
            fieldsToMerge = {'timetrain', 'etrain', 'yhat_train'};
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
                newEntry = temp.(dir).(fieldName).(dir);
                results.(fieldName){run}.DM{end+1} = newEntry{1};
            end

            % update D, M, counters, TN for next iteration of while-loop
            if dir == 'D'
                D_idx = D_idx + 1;
                D = D_vec(D_idx);
                TN = temp.D.TN_train.D{1};
                new_RMSE = RMSE_D;
                res = y_train - temp.D.yhat_train.D{1};
                yhat_train_old = yhat_train_D;
            elseif dir == 'M'
                M_idx = M_idx + 1;
                M = M_vec(M_idx);
                TN = temp.M.TN.M{1};
                new_RMSE = RMSE_M;
                res = y_train - temp.M.yhat_train.M{1};
                yhat_train_old = yhat_train_M;
            end

            % Check if error increased
            if new_RMSE > previous_relValError
                increase_count = increase_count + 1;
            else
                increase_count = 0;  % reset if no increase
            end

            % Update for next iteration
            previous_relValError = new_RMSE
            results.D_value{run}.DM{end+1} = D;
            results.M_value{run}.DM{end+1} = M;
            results.R_value{run}.DM{end+1} = R;
            counter = counter + 1;
        end
    end

end
% Save to file
% save(sprintf('results/experiment_increaseDM/results_DM_%s.mat', filenameTag), 'results');
end

function chosenDirection = chooseDirection(canIncrease_D, canIncrease_M, relValError_D, relValError_M)

    % All directions and their data
    directions = {'D', 'M'};
    canIncrease = [canIncrease_D, canIncrease_M];
    relValErrors = [relValError_D, relValError_M];

    % Set val error to Inf if that direction can't be increased
    relValErrors(~canIncrease) = -Inf;

    % Find direction with minimum relValError (among allowed ones)
    % [~, idx] = min(relValErrors);
    [~, idx] = max(relValErrors);

    if isinf(relValErrors(idx))
        chosenDirection = 'none';
        warning('No directions can be increased.');
    else
        chosenDirection = directions{idx};
    end

end

