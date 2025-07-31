function const = const_AAO_charging(a,t)

emin = repelem(0,2*t);
emax = repelem(0,2*t);
Qmax = a.InitialSoC;
for j = 1:t
    Pmax = min(a.Qmax-Qmax,a.L*a.Pmax);
    Qmax = Qmax + Pmax;
    emax(j) = Pmax/a.L;
end
Qmin = a.Qmax;
for j = t+1:2*t
    Pmin = max(a.Qmin-Qmin,a.Pmin);
    Qmin = Qmin + Pmin;
    emin(j) = a.L*Pmin;
end
const = [emin;emax];