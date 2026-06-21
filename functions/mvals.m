function [TN,e,yhat_train, TNinc, rndseed]=mvals(u,y,M,r,MAXITR,modeReg,strategy, varargin)
% [TN,e]=mvals(y,u,M,r) or [TN,e]=mvals(y,u,M,r,THRESHOLD,MAXITR)
% ---------------------------------------------------------------
% MIMO Volterra Alternating Linear Scheme (MVALS) algorithm for
% solving the MIMO Volterra system identification problem in the Tensor
% Network format.
%
% TN        =   Tensor Network, TN.core is a cell containing the TN-cores,
%               TN.n is a matrix where TN.n(i,:) are the dimensions of the
%               ith TN-core, TN.norm stores the location of the normcore,
%
% e         =   table containing additional information about results
%
% y         =   matrix, y(:,k) contains the kth output,
%
% u         =   matrix, u(:,k) contains the kth input,
%
% M         =   scalar, memory of each of the Volterra kernels,
%
% r         =   vector, contains the TT ranks r_1 up to r_{d-1}, since
%               r_0=r_d=1.
%
% MAXITR    =   scalar, optional maximum number of iterations. Default=100,
%
% modeReg   =   'reg' (for regularized update) or 'bat17' (for
%               unregularized update),
%
% strategy  =   'inc0' (for standard ALS), 'incM' (to increase M' or 'incD'
%               to increase D
%
% varargin  =   additional inputs (depending on modeReg and modeInit)
%               * If modeReg = 'reg'  and modeInit ≠ 'inc0' → {lambda_inv, sigma_sq, TNinit}
%               * If modeReg = 'bat17' and modeInit ≠ 'inc0' → {TNinit}
%               * If modeInit = 'incM'  → {TNinit} (must satisfy TNinit.n(1,3) < M)
%               * If modeInit = 'incD'  → {TNinit} (must satisfy dinit = length(r) + 2)
%               * If modeInit = 'inc0'  → No additional inputs

% Reference
% ---------
%
% 06/07/11 - 2016, Kim Batselier
% 15/03/25 - Eva Memmel

%% Handling Inputs
% Validate inputs
validModesReg = {'reg', 'bat17'};
validStrategy = {'inc0', 'inc0seed', 'incM', 'incD','linear'};

if ~ismember(modeReg, validModesReg)
    error('Invalid modeReg. Choose ''reg'' or ''bat17''.');
end
if ~ismember(strategy, validStrategy)
    error('Invalid modeInit. Choose ''inc0'', ''incM'', or ''incD''.');
end

% Ensure M and MAXITR are scalars
if ~isnumeric(M) || ~isscalar(M)
    error('M must be a numeric scalar.');
end
if ~isnumeric(MAXITR) || ~isscalar(MAXITR)
    error('MAXITR must be a numeric scalar.');
end

% Ensure r is a numeric array with at least one entry
if ~isnumeric(r) || isempty(r)
    error('r must be a numeric array with at least one entry.');
end

% Determine expected varargin length
argsReg = 0;
argsInit = 0; 

if strcmp(modeReg, 'reg')
    argsReg = 2; %Requires two extra arguments: lambda_inv, sigma_sq
end
% 'bat17' does not require any extra arguments (argsReg = 0)

if strcmp(strategy, 'incM')
    argsInit = 1; % incM requires TNinit
elseif strcmp(strategy, 'incD')
    argsInit = 1; % incD requires TNinit
elseif strcmp(strategy,'inc0seed')
    argsInit = 1; % inc0seed requires random seed
elseif strcmp(strategy,'linear')
    argsInit = 1;
end
% 'inc0' does not require any additional input (argsInit = 0)

expectedArgs = argsReg + argsInit;

% Check varargin length
if length(varargin) ~= expectedArgs
    error('Incorrect number of optional arguments. Expected %d, got %d.', expectedArgs, length(varargin));
end

index = 1;
% Handle modeReg-specific arguments
if argsReg == 2
    lambda_inv = varargin{index};
    sigma_sq = varargin{index + 1};

    if ~isnumeric(lambda_inv) || ~isnumeric(sigma_sq)
        error('lambda_inv and sigma_sq must be numeric.');
    end
    index = index + 2;
end

% Handle modeInit-specific arguments (TNinit for both incD and incM)
if strcmp(strategy, 'incM') || strcmp(strategy,'incD')
    TNinit = varargin{index};
    % Validate TNinit struct and check required fields inline
    if ~isstruct(TNinit) || ~all(isfield(TNinit, {'norm', 'n', 'core'}))
        error('TNinit must be a struct containing the fields: ''norm'', ''n'', and ''core''.');
    end
    
    p=size(u,2);                    % number of inputs
    dinit = size(TNinit.n,1);       % order of TNnit
    
    % Check condition for incM: TNinit.n(1,3) < M
    if strcmp(strategy, 'incM') && TNinit.n(1,3) >= p*M+1
        error('For incM, p*M+1 of initial TNinit must be less than p*M+1 for TN.');
    end
    
    % Check condition for incD: dinit = size(TNinit.n,1) and length(r) + 2
    if strcmp(strategy, 'incD') && dinit ~= length(r)
        error('For incD, we increase the order by one.');
    end
    
    % If norm is not 2, adjust it
    if TNinit.norm ~= 2
        rinit = [TNinit.n(:,1);1]';
        n = TNinit.n(1,3); %assuming uniform n
        siteK(2);
    end
end

if strcmp(strategy, 'inc0seed')
     rndseed = varargin{1};
end

if strcmp(strategy, 'linear')
    N_train = varargin{1};
end

%% Setting up variables
p=size(u,2);                    % number of inputs
[N,l]=size(y);                  % length, number of outputs
y=reshape(y',[N*l,1]);
d=length(r)+1;                  % degree of truncated Volterra series
r=[l r(:)' 1];                  % append extremal TN ranks
n=p*M+1;
THRESHOLD=1e-15;                % numeric-zero as stopping criterion
itr=1;                          % counts number of iterations
ltr=1;                          % flag that checks whether we sweep left to right

% Initialize e-table
e = table(zeros(MAXITR, 1), zeros(MAXITR, 1), zeros(MAXITR, 1), zeros(MAXITR, 1),zeros(MAXITR, 1),...
          zeros(MAXITR, 1), zeros(MAXITR, 1), zeros(MAXITR, 1), zeros(MAXITR, 1), zeros(MAXITR, 1),...
          'VariableNames', {'err_rel_train', 'norm_wd', 'norm_Ud', 'rank_wd_IR2', 'sweepindex', 'd', 'rank_A', 'orth_train', 'ratio_train', 'RMSE_train'});
e(1, :) = {1, 1, 1, 1, 1, 1, 1, 1, 1,1};

% construct N x n matrix U
U=zeros(N,n);
u=[zeros(M-1,p);u];
for i=M:N+M-1
    temp=ones(1,n);
    for j=1:M
        temp(2+(j-1)*p:2+j*p-1)=u(i-j+1,:);
    end
    U(i-M+1,:)=temp;
end
u=u(M:end,:);
Vp=cell(1,d);
Vm=cell(1,d);
if l==1
    Vm{1}=ones(N,1);
else
    Vm{1}=eye(l);
end
Vp{d}=ones(N,1);

%% Initializing TT
TN.core=cell(1,d);
TN.n(1,:)=[1 l n r(2)];
if strcmp(strategy, 'incD')
    % Increase D from D-1 to D by creating a new core that only contains
    % the norm
    TN.core{1} = TNinit.core{1};
    for i = d:-1:4
        TN.n(i,:)=[r(i) 1 n r(i+1)];
        % TN.core{i} = TNinit.core{i-1};
        ninit = TNinit.n(i-1,:);
        TN.core{i}(1:ninit(1),1:ninit(3),1:ninit(4)) = TNinit.core{i-1};
        Vp{i-1}=dotkron(Vp{i},U)*reshape(permute(TN.core{i},[3 2 1]),[r(i+1)*n,r(i)]); % N x r_{i-1}
    end
    TN.n(3,:)=[r(3) 1 n r(4)];
    [U2,S2, Vt2] = svd(reshape(TNinit.core{2}(:),[r(3),(n)*r(4)])','econ');
    TN.core{3}=reshape(U2',[r(3),n,r(4)]);
    TN.core{1}=reshape(reshape(TNinit.core{1},[l*n,r(2)])*Vt2,[l,n,r(2)]);
    TN.n(2,:)=[r(2) 1 n r(3)];
    TN.core{2}=zeros([r(2) n r(3)]);
    TN.core{2}(:,1,:) = S2;
    Vp{2}=dotkron(Vp{3},U)*reshape(permute(TN.core{3},[3 2 1]),[r(4)*n,r(3)]); % N x r_{i-1}
    TN.n(1,:)=[1 l n r(2)];
    % % Alternative with QR decomposition
    % TN.n(3,:)=[r(3) 1 n r(4)];
    % [Q,R]=qr(reshape(TNinit.core{2}(:),[r(3),(n)*r(4)])');
    % TN.core{3}=reshape(Q(:,1:r(3))',[r(3),n,r(4)]);
    if l==1
        Vm{2}=dotkron(Vm{1},U)*reshape(TN.core{1},[r(1)*n,r(2)]); % N x r_{i}
    else
        Vm{2}=U*reshape(permute(TN.core{1},[2 1 3]),[n,r(1)*r(2)]); %N x r_{i-1}r_i
        Vm{2}=reshape(Vm{2},[N,r(1)*r(2)]);
    end
    sweepindex=2; % index that indicates which TT core will be updated
    % ltr = 0;

   

elseif strcmp(strategy,'linear')
    w = pinv(U(1:N_train,:))*y(1:N_train);
    [Qw,Rw] = qr(w,'econ');
    TN.core{1} = Qw;
    TN.core{2} = Rw.*[1; zeros(n-1,1)];
    TN.n(1,:) = [1 1 n 1];
    TN.n(2,:) = [1 1 n 1];
    sweepindex = 2; ltr = 0;
    Vm{2}=dotkron(Vm{1},U)*reshape(TN.core{1},[r(1)*n,r(2)]);
    

elseif strcmp(strategy, 'incM')
    % Increasing M by appending zeroes of size Mplus
    TN.core{1} = zeros(1,n,r(2));
    ninit = TNinit.n(1,:);
    TN.core{1}(1:ninit(1),1:ninit(3),1:ninit(4)) = TNinit.core{1};
    if l==1
        Vm{2}=dotkron(Vm{1},U)*reshape(TN.core{1},[r(1)*n,r(2)]); % N x r_{i}
    else
        Vm{2}=U*reshape(permute(TN.core{1},[2 1 3]),[n,r(1)*r(2)]); %N x r_{i-1}r_i
        Vm{2}=reshape(Vm{2},[N,r(1)*r(1)]);
    end
    for i = d:-1:2
        TN.n(i,:)=[r(i) 1 n r(i+1)];
        TN.core{i} = zeros(r(i), n, r(i+1));
        ninit = TNinit.n(i,:);
        TN.core{i}(1:ninit(1),1:ninit(3),1:ninit(4)) = TNinit.core{i};
        Vp{i-1}=dotkron(Vp{i},U)*reshape(permute(TN.core{i},[3 2 1]),[r(i+1)*n,r(i)]); % N x r_{i-1}
    end
    sweepindex=2; % index that indicates which TT core will be updated
    if d ==2
        ltr = 0;
    end
else
    if strcmp(strategy,'inc0')
        rng('shuffle');
        t = rng;
        rndseed = t.Seed ;
    end
    TN.core{1}=rand(r(1),n,r(2));
    TN.core{1}=TN.core{1}./norm(TN.core{1}(:));
    for i=d:-1:2
        TN.n(i,:)=[r(i) 1 n r(i+1)];
        TN.core{i}=permute(reshape(orth(rand((n)*r(i+1),r(i))),[r(i+1),(n),r(i)]),[3,2,1]);
        TN.core{i}=TN.core{i}./norm(TN.core{i}(:));
        Vp{i-1}=dotkron(Vp{i},U)*reshape(permute(TN.core{i},[3 2 1]),[r(i+1)*n,r(i)]); % N x r_{i-1}
    end
    sweepindex = 1; % index that indicates which TT core will be updated
end

%% Looping Sweeps

while (itr < MAXITR) && e{itr,1} > THRESHOLD
    updateTT;
    updatesweep;
end


%% UpdateTT
    function updateTT
        % first construct the linear subsystem matrix
        if l==1
            A=dotkron(Vm{sweepindex},U,Vp{sweepindex});
        elseif sweepindex == 1
            A=kron(dotkron(U,Vp{sweepindex}),Vm{sweepindex});
        else
            A=dotkron(Vm{sweepindex},U,Vp{sweepindex});
            A=reshape(A,[N,l,r(sweepindex)*n*r(sweepindex+1)]);
            A=permute(A,[2 1 3]);
            A=reshape(A,[N*l,r(sweepindex)*n*r(sweepindex+1)]);
        end
        if itr == 1
            y_hat_old = A*TN.core{sweepindex}(:);
            TNinc = TN;
        end
        if strcmp(modeReg, 'bat17')
            g=pinv(A)*y;
        else
            I = eye(size(A,2));
            g = (lambda_inv*I + A'*A/sigma_sq)\(A'*y/sigma_sq);
        end
        yhat_train = A*g;
        err_rel_train = norm(yhat_train(l*M+1:end)-y(l*M+1:end))/norm(y(l*M+1:end));
        norm_wd = norm(g);
        norm_Ud = norm(A(:));
        rank_A = rank(A);
        rank_wd_IR2 = rank(reshape(permute(g,[2 1 3]),[n,r(sweepindex)*r(sweepindex+1)]));  
        RMSE_train = rmse(yhat_train(l*M+1:end),y(l*M+1:end));
        if itr == 1
            orth_train = (yhat_train-y_hat_old)'*y_hat_old;
            ratio_train = (yhat_train-y_hat_old)'*y/((yhat_train-y_hat_old)'*(yhat_train-y_hat_old));
        else
            orth_train = NaN;
            ratio_train = NaN;
        end
        itr=itr+1;
        e(itr, :) = {err_rel_train, norm_wd, norm_Ud, rank_wd_IR2, sweepindex, d, rank_A, orth_train, ratio_train, RMSE_train}; 

        if ltr
            % left-to-right sweep, generate left orthogonal cores and update vk1
            [Q,R]=qr(reshape(g,[r(sweepindex)*(n),r(sweepindex+1)]));
            TN.core{sweepindex}=reshape(Q(:,1:r(sweepindex+1)),[r(sweepindex),n,r(sweepindex+1)]);
            TN.core{sweepindex+1}=reshape(R(1:r(sweepindex+1),:)*reshape(TN.core{sweepindex+1},[r(sweepindex+1),(n)*r(sweepindex+2)]),[r(sweepindex+1),n,r(sweepindex+2)]);
            if l==1
                Vm{sweepindex+1}=dotkron(Vm{sweepindex},U)*reshape(TN.core{sweepindex},[r(sweepindex)*n,r(sweepindex+1)]); % N x r_{i}
            elseif sweepindex==1
                Vm{sweepindex+1}=U*reshape(permute(TN.core{sweepindex},[2 1 3]),[n,r(sweepindex)*r(sweepindex+1)]); %N x r_{i-1}r_i
                Vm{sweepindex+1}=reshape(Vm{sweepindex+1},[N,r(sweepindex)*r(sweepindex+1)]);
            else
                Vm{sweepindex+1}=reshape(dotkron(Vm{sweepindex},U),[N*l,r(sweepindex)*n])*reshape(TN.core{sweepindex},[r(sweepindex)*n,r(sweepindex+1)]);
                Vm{sweepindex+1}=reshape(Vm{sweepindex+1},[N,l*r(sweepindex+1)]);
            end
        else
            % right-to-left sweep, generate right orthogonal cores and update vk2
            [Q,R]=qr(reshape(g,[r(sweepindex),(n)*r(sweepindex+1)])');
            TN.core{sweepindex}=reshape(Q(:,1:r(sweepindex))',[r(sweepindex),n,r(sweepindex+1)]);
            TN.core{sweepindex-1}=reshape(reshape(TN.core{sweepindex-1},[r(sweepindex-1)*(n),r(sweepindex)])*R(1:r(sweepindex),:)',[r(sweepindex-1),n,r(sweepindex)]);
            Vp{sweepindex-1}=dotkron(Vp{sweepindex},U)*reshape(permute(TN.core{sweepindex},[3 2 1]),[r(sweepindex+1)*n,r(sweepindex)]); % N x r_{i-1}
        end
    end


%% UpdateSweep
    function updatesweep
        %display(num2str(sweepindex));
        if ltr
            sweepindex=sweepindex+1;
            if sweepindex== d
                ltr=0;
            end
        else
            sweepindex=sweepindex-1;
            if sweepindex== 1
                ltr=1;
            end
        end
    end
%% SiteK
    function siteK(K)
        for k = 1:K-1
            [Q,R]=qr(reshape(TNinit.core{k},[rinit(k)*(n),rinit(k+1)]));
            TNinit.core{k}=reshape(Q(:,1:rinit(k+1)),[rinit(k),n,rinit(k+1)]);
            TNinit.core{k+1}=reshape(R(1:rinit(k+1),:)*reshape(TNinit.core{k+1},[rinit(k+1),(n)*rinit(k+2)]),[rinit(k+1),n,rinit(k+2)]);
        end
        for k = dinit:-1:K+1
            [Q,R]=qr(reshape(TNinit.core{k},[rinit(k),(n)*rinit(k+1)])');
            TNinit.core{k}=reshape(Q(:,1:rinit(k))',[rinit(k),n,rinit(k+1)]);
            TNinit.core{k-1}=reshape(reshape(TNinit.core{k-1},[rinit(k-1)*(n),rinit(k)])*R(1:rinit(k),:)',[rinit(k-1),n,rinit(k)]);
        end
        TNinit.norm = K;
    end
%%
% Trim table if fewer iterations were run
e = e(1:itr, :);
TN.norm = sweepindex;
end
