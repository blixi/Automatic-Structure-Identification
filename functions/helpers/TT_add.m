function sum  = TT_add(A,B)
% Author: Eva Marie Memmel
% e.m.memmel@tudelft.nl
%%
dim = size(A.n, 1);  % Number of cores
sum.core = cell(1, dim);
sum.n = zeros(dim, 4);
sum.norm = 0;

% First core
sum.core{1} = cat(3, A.core{1}, B.core{1});
sum.n(1,1) = A.n(1,1);                             % Left rank
sum.n(1,2) = A.n(1,2);                              % L
sum.n(1,3) = A.n(1,3);                              % Mode size
sum.n(1,4) = A.n(1,4) + B.n(1,4);                   % New right rank

% Middle cores
for i = 2:dim-1
    r1 = A.n(i,1); r2 = A.n(i,4);
    s1 = B.n(i,1); s2 = B.n(i,4);
    n = A.n(i,3); % mode size

    C = zeros(r1 + s1, n, r2 + s2);

    % Fill top-left block with A
    C(1:r1, :, 1:r2) = A.core{i};

    % Fill bottom-right block with B
    C(r1+1:end, :, r2+1:end) = B.core{i};

    sum.core{i} = C;
    sum.n(i,:) = [r1 + s1, 1, n, r2 + s2];
end

% Last core (concatenate vertically)
sum.core{dim} = cat(1, A.core{dim}, B.core{dim});
sum.n(dim,1) = A.n(dim,1) + B.n(dim,1);           % New left rank
sum.n(dim,2) = 1;
sum.n(dim,3) = A.n(dim,3);
sum.n(dim,4) = A.n(dim,4);                        % Right rank (should be 1) 
end

