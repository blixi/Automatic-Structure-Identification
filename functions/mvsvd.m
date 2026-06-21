function [QTN,e,yhat_train] = mvsvd(u,y,M,D,modeRound)
% Compute mvsvd based on Kim Batselier
% https://github.com/kbatseli/SymmetricVolterra/tree/master)
% Inputs:
%   - u:                N x P input
%   - y:                N x L output
%   - M:                chosen lag of Volterra model
%   - D:                chosen order of Volterra model
% Outputs:
%   - QTN:              TN computed with mvsvd

% 16/03/25 - Eva Memmel

%% Setting up variables
if strcmp(modeRound.method,'rank')
    modeRound.tol = modeRound.R*ones(1,D-1);
end
QTN.core=cell(1,D);
p=size(u,2);                    % number of inputs
[N,L]=size(y);                  % length, number of outputs
n=p*M+1;
U=makeU(u,M,1);
%% Compute SVD
rsvd = [arrayfun(@(d) nchoosek(d+n-1, n-1), D:-1:2),n, 1];
for d=D:-1:2
    if d==D
        [V,S,Qt]=svd(U,'econ');
        QTN.n(D,1:4) = [n,1,n,1];
        QTN.core{d} = reshape(Qt',[n,n]);
    else
        [V,S,Qt]=svd(reshape(dotkron(U,V*S),[N,rsvd(d+1)*n]),'econ');
        QTN.n(d,1:4) = [rsvd(d),1,n,rsvd(d+1)];
        QTN.core{d} = reshape(Qt(:,1:rsvd(d))',[rsvd(d),n,rsvd(d+1)]);
        S=S(1:rsvd(d),1:rsvd(d));
        V=V(:,1:rsvd(d));
    end
end
[V,S,Qt]=svd(reshape(dotkron(U,V*S),[N,rsvd(2)*n]),'econ');
Qt = Qt(:,1:rsvd(1));
S=S(1:rsvd(1),1:rsvd(1));
V=V(:,1:rsvd(1));
QTN.core{1} = reshape(y'*V*pinv(S)*Qt',[L,n,rsvd(2)]);
QTN.n(1,(1:4)) = [1,L,n,rsvd(2)];
QTN.norm = 1;

%% Perform rounding and compute errors
[QTN,eround] = roundTN(QTN,modeRound.tol,modeRound.method);
yhat_train = sim_volterraTN(u,QTN);
err_rel_train = relative_error(yhat_train(L*M+1:end),y(L*M+1:end));
RMSE_train = rmse(yhat_train(L*M+1:end),y(L*M+1:end));
norm_wd = norm(QTN.core{QTN.norm}(:));
e = table(err_rel_train,eround,norm_wd,RMSE_train,'VariableNames',{'err_rel_train','err_rel_round','norm_wd','RMSE_train'});
end

