function const = const_prob_charging(slices1,slices2,space,t)
%TO BE REWORKED
ss1 = size(slices1);
s1 = ss1(1);
ss2 = size(slices2);
s2 = ss2(1);
emin = repelem(0,s1+s2);
emax = repelem(0,s1+s2);
for j = 1:s1
    v = find(slices1(j,:) >= t);
    s11 = size(v);
    s = s11(2);
    emin(j) = 0;
    emax(j) = space(v(s));
end
for j = 1:s2
    v = find(slices2(j,:) >= t);
    s22 = size(v);
    s = s22(2);
    emin(j+s1) = space(v(1));
    emax(j+s1) = 0;
end
const = [emin;emax];