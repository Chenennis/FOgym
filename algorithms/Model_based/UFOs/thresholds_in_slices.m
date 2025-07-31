function MR = thresholds_in_slices(slice,p)

%extract and initialize
slices = slice.slices;
space = slice.space;
ss = size(slices);
T = ss(1);
MR = repelem(0,2,T);

%elaborate thresholds
for t = 1:T
    sl = slices(t,:);
    ssp = size(space);
    lastelem = ssp(2);
    if sl(1) > p
        MR(1,t) = space(1);
    else
        th1 = find(sl >= p,1);
        conv = (sl(th1)-p)/(sl(th1)-sl(th1-1));
        MR(1,t) = conv*space(th1-1) + (1-conv)*space(th1);
    end
    if sl(lastelem) > p
        MR(2,t) = space(lastelem);
    else
        th2 = find(sl >= p,1,'last');
        conv = (sl(th2)-p)/(sl(th2+1)-sl(th2));
        MR(2,t) = conv*space(th2) + (1-conv)*space(th2+1);
    end
end