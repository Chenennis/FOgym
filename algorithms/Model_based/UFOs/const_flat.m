function const = const_flat(a,t)

emin = repelem(0,t);
emax = repelem(0,t);
Qmin = a.L*(a.Qmin - a.InitialSoC);
Qmax = (a.Qmax - a.InitialSoC)/a.L;
for j = 1:t
    emin(j) = Qmin/t;
    emax(j) = Qmax/t;
end
const = [emin;emax];