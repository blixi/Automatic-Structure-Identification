function TN_transposed = transposeTN(TN)
% transpose a matrix/vector in tensor network form

for i=1:size(TN.n,1)
   TN_transposed.core{i} = permute(TN.core{i},[1,3,2,4]);
   TN_transposed.n(i,:) = [TN.n(i,1),TN.n(i,3),TN.n(i,2),TN.n(i,4)];
end

end