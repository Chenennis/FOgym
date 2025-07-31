function const = const_flat_charging(a,t)

emin = repelem(0,2*t);
emax = repelem(0,2*t);
Qmax = (a.Qmax - a.InitialSoC)/a.L;
for j = 1:t
    emax(j) = Qmax/t;
end
for j = t+1:2*t
    emin(j) = a.L*(a.Qmin - a.Qmax)/t;
end
const = [emin;emax];