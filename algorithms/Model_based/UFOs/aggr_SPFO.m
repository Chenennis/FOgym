function E = aggr_SPFO(PFO,pt)

%unpack number of devices and time
s = size(PFO);
num = s(2);
sl1 = PFO(1).slices;
ss = size(sl1);
T = ss(1);

% initialize constraints
E = repelem(0,2,T);
p = nthroot(pt,num);

% build constraints
for k = 1:num
    MR = thresholds_in_slices(PFO(k),p);
    E = E + MR;
end



