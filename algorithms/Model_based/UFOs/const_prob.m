function const = const_prob(slices,space,t)

ss1 = size(slices);
s1 = ss1(1);
emin = repelem(0,s1);
emax = repelem(0,s1);
for j = 1:s1
    v = find(slices(j,:) >= t);
    s2 = size(v);
    s = s2(2);
    emin(j) = space(v(1));
    emax(j) = space(v(s));
end
const = [emin;emax];