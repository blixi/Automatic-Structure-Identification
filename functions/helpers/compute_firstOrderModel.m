function firstVKM = compute_firstOrderModel(TN)
d = size(TN.n,1);
n = TN.n(1,3);
L = TN.n(1,2);

firstVKM = zeros(L*n,1);
for i = 1:d
    % disp("====== i ======")
    % disp(i)
    if i == 1
        b=reshape(TN.core{1},[prod(TN.n(1,1:3)) TN.n(1,end)]);
    else
        if L == 1
            b = reshape(TN.core{1}(:,1,:),[prod([TN.n(1,1),TN.n(1,2)]) TN.n(1,end)]);
        else
            b = reshape(TN.core{1}(:,1:L,1,:),[prod([TN.n(1,1),TN.n(1,2)]) TN.n(1,end)]); % L x R
        end
    end
    for j = 2:d
        % disp('-----j------')
        % disp(j)
        if j ~= i
            zerothOK = reshape(TN.core{j}(:,1,:),[TN.n(j,1) prod([TN.n(j,2),TN.n(j,end)])]); % R x R
            b = b*zerothOK; % L x R
        else
            firstOK = reshape(TN.core{i},[TN.n(i,1) prod(TN.n(i,2:end))]); % R x IR
            b = b*firstOK; % L x R times R x IR = L x IR
            b = reshape(b,[L*n,TN.n(i,end)]); % LI x R
        end
    end
    firstVKM = firstVKM + b; % LxI
end
firstVKM = reshape(firstVKM,[L,n]);
end

