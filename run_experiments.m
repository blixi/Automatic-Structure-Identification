% Main Experiment File
clear all; close all; clc;

% Add all relevant folders to path
currentFolder = fileparts(mfilename('fullpath'));
parentFolder = fullfile(currentFolder);
addpath(genpath(currentFolder));

% Is going to be set true in corresponding parameter file
runIncrease0 = false;
runIncreaseD = false;
runIncreaseM = false;
runIncreaseDM = false;
runWatertanks = false;
runGridSearchDS = false;
runGridSearchDSOTA = false;


%% Parameters Experiment Increase D

% Experiment 01: Dataset: D = 7, M = 5, R = 4; Noise-Free
% goal: Show how increaseD behaves if everything else is chosen as "true"
% run('experiment01.m')

%% Parameters Experiment Increase DM

% Experiment DM 01 D = 7, M = 5, R = 4
% run('experimentDM01.m')

run('watertank_grid_search_D_sweeps.m')

% run('watertank_config')

%%

% Store Info in results
results = struct();
results.info = struct('dataset', datasetName, 'date', dateString, 'N_train', N_train, 'N_val', N_val, 'Dtrue', dtrue, 'Mtrue', Mtrue,...
    'R', R, 'MAXSWEEPS', MAXSWEEPS, 'totalMAXITR',totalMAXITR, 'D_vec', D_vec, 'M_vec', M_vec, 'D0', D0, 'D0M', D0M, 'M0', M0, 'M0D', M0D, ...
    'modeReg', modeReg, 'strategyInitALS', strategyInit.ALSinit, 'strategyInitSVD', strategyInit.SVDinit, 'runsALS', runs.runsALS, 'runsSVD', runs.runsSVD, 'evaluateOptions', evaluateOptions);
results.info.strategyField = strategyList;
%% Call Different Experiment Functions

% Run selected experiments
if runIncrease0
    results.increase0 = experiment_increase0(u_train, y_train, u_val, y_val, u_test, u_test, M0D, R, D0M, totalMAXITR, runs, modeReg, modeRound, strategyInit, evaluateOptions, filenameTag);
end

if runIncreaseD
    initialD = D0;
    initialM = M0D;
    results.increaseD = experiment_increaseD(u_train, y_train, u_val, y_val, u_test, y_test, initialM, R, initialD, D_vec, MAXSWEEPS, totalMAXITR, runs, modeReg, modeRound, strategyInit, evaluateOptions, filenameTag);
end

if runIncreaseM
    initialD = D0M;
    initialM = M0;
    results.increaseM = experiment_increaseM(u_train, y_train, u_val, y_val, u_test, y_test, initialM, M_vec, R, initialD, MAXSWEEPS, totalMAXITR, runs, modeReg, modeRound, strategyInit, evaluateOptions, filenameTag);
end

if runIncreaseDM
    results.increaseDM = experiment_increaseDM(u_train, y_train, u_val, y_val, u_test, y_test, M_vec, D_vec, R, MAXSWEEPS, runs, modeReg, modeRound, strategyInit, evaluateOptions, filenameTag);
end

if runWatertanks
    results.increaseD = train_1_watertanks(u_train, y_train, u_val, y_val, u_test, y_test, M_vec, D_vec, R, MAXSWEEPS, runs, modeReg, modeRound, strategyInit, evaluateOptions, filenameTag);
    % [D_vec_chosen, M_vec_chosen] = determine_model_order_SVD(results);
    % idxSVD = find(strcmp(results.determineOrder.method, 'SVDinit'),1);
    % results.finalModelSVD = train_2_watertanks(u_train, y_train, u_val, y_val, u_test, y_test, M_vec_chosen, D_vec_chosen, R, MAXSWEEPS, runs, modeReg, modeRound, strategyInit, evaluateOptions, filenameTag, results.determineOrder.updated{idxSVD});
    % time_config = learning_rate_timing(results);
    % [D_chosen, M_chosen, distribution] = determine_model_order_SOTA(results);
    % results.finalModelSOTA = train_3_watertanks(u_train, y_train, u_val, y_val, u_test, y_test, M_chosen, D_chosen, R, MAXSWEEPS, runs, modeReg, strategyInit, evaluateOptions);
    % results.finalModelSOTA.distribution = distribution;
end

if runGridSearchDS
    results.gridDS = performGridSearchDS(u_train, y_train, u_val, y_val, u_test, y_test, M_vec, D_vec, R, MAXSWEEPS, runs, modeReg, modeRound, strategyInit, evaluateOptions, filenameTag)
end

if runGridSearchDSOTA
    results.gridDSOTA = performGridSearchDSOTA(u_train, y_train, u_val, y_val, u_test, y_test, M_vec, D_vec, R, MAXITR, runs, modeReg, modeRound, strategyInit, evaluateOptions, filenameTag)
end

% Save results
save(sprintf('results/results_%s.mat', filenameTag), 'results');

fprintf('Experiment completed and results saved as: results_%s.mat\n', filenameTag);

function time_config = learning_rate_timing(results)
time_train = 0;
time_val = 0;
for i = 1:length(results.determineOrder.timetrain{1}.DM)
    time_train = time_train + results.determineOrder.timetrain{1}.DM{i};
    time_val = time_val + results.determineOrder.timeval{1}.DM{i};
end
time_config = time_train + time_val;
end

function [D_vec_chosen, M_vec_chosen] = determine_model_order_SVD(results)
% idxSVD = find(strcmp(results.determineOrder.method, 'SVDinit'),1);
idxSVD = find(strcmp(results.determineOrder.method, 'ALSinit'));
% incMax = length(results.determineOrder.val{idxSVD}.DM);

% VAF_SVD = zeros(incMax,1);
RMSE_SVD = repmat(NaN,[15,20]);
% relError_SVD = zeros(incMax,1);
for j = 1:length(idxSVD)
    idx = idxSVD(j);
    incMax = length(results.determineOrder.val{idx}.DM);
    for i = 1:incMax
        % VAF_SVD(i,1) = results.determineOrder.VAF{idxSVD}.DM{i}.VAF_Abs;
        % relError_SVD(i,1) = results.determineOrder.val{idxSVD}.DM{i}.relError;
        RMSE_SVD(i,j) = results.determineOrder.val{idx}.DM{i}.RMSE;
    end
end

% decide based on RMSE
pos = find(RMSE_SVD == min(RMSE_SVD));
D_chosen = results.determineOrder.D_value{idxSVD}.DM{pos};
M_chosen = results.determineOrder.M_value{idxSVD}.DM{pos};

D_cutoff = find(results.info.D_vec == D_chosen);
M_cutoff = find(results.info.M_vec == M_chosen);

D_vec_chosen = results.info.D_vec(1:D_cutoff);
M_vec_chosen = results.info.M_vec(1:M_cutoff);

end

function [D_chosen, M_chosen, distribution] = determine_model_order_SOTA(results)
idxSOTA = find(strcmp(results.increaseD.method, 'SOTA'));
runs = length(idxSOTA);
min_RMSE = zeros(runs,1);
DM_valueSOTA = zeros(runs,2);
posGridRMSE = zeros(runs,1);
for run = 1:runs
    idx = idxSOTA(run);
    min_RMSE(run,1) = results.increaseD.val{idx}.D{1}.RMSE;
    DM_valueSOTA(run,:) = [results.increaseD.D_value{idx}.D{1} results.increaseD.M_value{idx}.D{1}];
    posGridRMSE(run,1) = 1;
    for i = 1:length(results.increaseD.val{idx}.D)
        if results.increaseD.val{idx}.D{i}.RMSE < min_RMSE(run,1)
            min_RMSE(run,1) = results.increaseD.val{idx}.D{i}.RMSE;
            DM_valueSOTA(run,:) = [results.increaseD.D_value{idx}.D{i} results.increaseD.M_value{idx}.D{i}];
            posGridRMSE(run,1) = i;
        end
    end


end

[distribution.GC,distribution.GR,distribution.GP] = groupcounts(posGridRMSE);
posGrid = distribution.GR(find(distribution.GC == max(distribution.GC)))
D_chosen = results.increaseD.D_value{idxSOTA(end)}.D{posGrid}
M_chosen = results.increaseD.M_value{idxSOTA(end)}.D{posGrid}

end
