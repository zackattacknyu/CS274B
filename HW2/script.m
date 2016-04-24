adj = zeros(5,5);
adj(1,2)=1;
adj(2,3)=1;
adj(4,5)=1;
adj = adj+adj';

adjTotal = adj;
adjPower = adj;
for i = 1:size(adj,1)
    adjPower = adjPower*adj;
    adjTotal = adjTotal + adjPower;
end
adjTotal(adjTotal>0)=1