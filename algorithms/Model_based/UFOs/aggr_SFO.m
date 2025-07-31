function const = aggr_SFO(models)

s = size(models);
s1 = s(1);
s2 = s(2);
u = floor(s1/2);
emin = repelem(0,s2);
emax = repelem(0,s2);
for t = 1:s2
    emin(t) = sum(models([1:2:s1],t));
    emax(t) = sum(models([2:2:s1],t));
end
const = [emin;emax];