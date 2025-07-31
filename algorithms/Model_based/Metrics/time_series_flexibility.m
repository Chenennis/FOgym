function TSF = time_series_flexibility(FO,TF)

if ~exist('TF','var')
    TF = 0;
end
slices = FO.slices;
length = size(slices);
k = length(1);
minA = repelem(0,k+TF); %initialize minA and maxA
maxA = repelem(0,k+TF);

for t = 1:k
   slice = slices(t);
   minA(t) = ymin(slice);
   maxA(t+TF) = ymax(slice);
end
TSF = norm(maxA-minA);
end