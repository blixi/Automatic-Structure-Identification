% Sept. 2025 - Eva Memmel
% ------------------------------------------

% Load dataset
datasetName = 'dataBenchmark';     
dateString = datestr(now, 'yyyy-mm-dd_HH-MM');
filenameTag = sprintf('%s_%s', datasetName, dateString);
load(sprintf('datasets/raw/%s.mat', datasetName));

% Training Validation Split (ca 2:1) for given training data (N = 1024)
% N_train = 682;
N_train = 682;
N_val = 1024-N_train;
N_test = 1024;

u_train = uEst(1:N_train); 
y_train = yEst(1:N_train);


y_val = yEst(N_train+1:end);
u_val = uEst(N_train+1:end);

u_test = uVal;
y_test = yVal;

% unknown true underlying values
dtrue = NaN;
Mtrue = NaN;

% Define search grid
D_vec = 2:15;
MAXSWEEPS = 1:10;


% Define remaining parameter
R = 1;
M_vec = [95];


% For info_vector/plots
D0 = D_vec(1);  % Initial D for increaseD
M0 = M_vec(1);  % Initial M for increaseM
D0M = D0; % Initial D for increase M
M0D = M0; % Initial M for increase D
totalMAXITR = NaN;

% Define number of runs 
% No random restarts for ALS, using option 'linear' for mvals
runsALS = 1;
runsSVD = 0;
runs = struct('runsALS',runsALS,'runsSVD',runsSVD);


runGridSearchDS = true;


% Define increase method
strategyList = {'increaseD'};

% Choose MVALS mode (reg vs bat17)
modeReg = 'bat17'; % Options: 'reg', 'bat17'

% Choose MVSVD mode (error + rank)
modeRound = struct('method','rank','R',R,'tol',[]); % Options: 'error', 'ranks'; tol=upper_bound, tol=rank_array

% Choose Initialization Strategy
strategyInit = struct('SOTA', false, 'ALSinit', true, 'SVDinit', false);

% Select Evaluation Mode
evaluateOptions = struct('ValidationData',true,'TestData',true, 'StoreTN',true);










