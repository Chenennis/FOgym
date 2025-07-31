function RAF = relative_area_flexibility(FO,TF)

if ~exist('TF','var')
    TF = 0;
end

slices = FO.slices;
length = size(slices);
k = length(1);
minA = repelem(0,k+TF); %initialize minA and maxA
maxA = repelem(0,k+TF);
avgA = repelem(0,k);

slice1 = slices(1);
max1 = ymax(slice1);

for t = 1:TF
    maxA(t) = max1;    
end

for t = 1:k
   slice = slices(t);
   ym = ymin(slice);
   YM = ymax(slice);
   minA(t) = ym;
   maxA(t+TF) = YM;
   avgA(t) = (YM+ym)/2;
end
AAF = sum(maxA) - sum(minA);
RAF = AAF/sum(avgA);
end