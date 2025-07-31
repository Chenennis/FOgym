function VF = advanced_vector_flexibility(FO,alpha,TF)

if ~exist('TF','var')
    TF = 0;
end

EF = energy_flexibility(FO);
vector = [alpha*TF,EF];
VF = norm(vector);

end