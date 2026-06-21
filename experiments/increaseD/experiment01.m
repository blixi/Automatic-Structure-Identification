% Sept. 2025 - Eva Memmel
% ------------------------------------------

% Set experiment parameters
datasetName = 'A41_Exp_Decay_D7_M5_rng1';      % Choose dataset to load

dateString = datestr(now, 'yyyy-mm-dd_HH-MM');
filenameTag = sprintf('%s_%s', datasetName, dateString);

% Load dataset
load(sprintf('datasets/raw/%s.mat', datasetName));

dtrue = 7;
Mtrue = 5;

N_train = 3000;
N_val = 750;
N_test = 750;

% Split training/Validation Data
u_train = u(1:N_train,:); u_val = u(N_train+1:N_train+N_val,:); u_test= u(N_train+N_val+1:end,:);
y_train = y(1:N_train,:); y_val = y(N_train+1:N_train+N_val,:); y_test= y(N_train+N_val+1:end,:);

% Define parameter settings
R = 3;
D_vec = 2:10;
M_vec = 3:7;
D0 = D_vec(1);  % Initial D for increaseD
M0 = M_vec(1);  % Initial M for increaseM

D0M = 2; % Initial D for increase M
M0D = 3; % Initial M for increase D

% Number of runs and sweeps (sweep is back and forth!)
MAXSWEEPS = 3;
totalMAXITR = sum(arrayfun(@(D) computeMAXITR(D, MAXSWEEPS), D_vec))-2*(1+D_vec(end)-D0M)+1; %initialALS starts at 1
% totalMAXITRincM = sum(arrayfun(@(i) computeMAXITR(D0M, MAXSWEEPS), 1:length(M_vec))) - length(M_vec)+1;
% Set total number of runs

runsALS = 1;
runsSVD = 1;
runs = struct('runsALS',runsALS,'runsSVD',runsSVD);

% Select Increase Strategies
runIncrease0 = true;
runIncreaseD = false;
runIncreaseM = false;
runIncreaseDM = true;

strategyList = {};
if runIncreaseD
    strategyList{end+1} = 'increaseD';
end
if runIncreaseM
    strategyList{end+1} = 'increaseM';
end
if runIncreaseDM
    strategyList{end+1} = 'increaseDM';
end

% Choose MVALS mode (reg vs bat17)
modeReg = 'bat17'; % Options: 'reg', 'bat17'

% Choose MVSVD mode (error + rank)
modeRound = struct('method','rank','R',R,'tol',[]); % Options: 'error', 'ranks'; tol=upper_bound, tol=rank_array

% Choose Initialization Strategy
strategyInit = struct('SOTA', true, 'ALSinit', true, 'SVDinit', true);

% Select Evaluation Mode
evaluateOptions = struct('ValidationData',true,'TestData',true, 'StoreTN',true);