function AAF = absolute_area_flexibility(FO,TF)

if ~exist('TF','var')
    TF = 0;
end

slices = FO.slices;
length = size(slices);
k = length(1);
minA = repelem(0,k+TF); %initialize minA and maxA
maxA = repelem(0,k+TF);

slice1 = slices(1);
max1 = ymax(slice1);

for t = 1:TF
    maxA(t) = max1;    
end

for t = 1:k
   slice = slices(t);
   minA(t) = ymin(slice);
   maxA(t+TF) = ymax(slice);
end
AAF = sum(maxA) - sum(minA);

end