function const = const_AAO(a,t)

emin = repelem(0,t);
emax = repelem(0,t);
Qmin = a.InitialSoC;
Qmax = a.InitialSoC;
for j = 1:t
    Pmin = max(a.Qmin-Qmin,a.Pmin);
    Pmax = min(a.Qmax-Qmax,a.L*a.Pmax);
    Qmin = Qmin + Pmin;
    Qmax = Qmax + Pmax;
    emin(j) = a.L*Pmin;
    emax(j) = Pmax/a.L;
end
const = [emin;emax];