function VF = vector_flexibility(FO,TF)

if ~exist('TF','var')
    TF = 0;
end
EF = energy_flexibility(FO);
vector = [TF,EF];
VF = norm(vector);

end
