function PFOresults = UFO_2steps(DFOEnergyValue,data,pt,prices,sett)

siz = size(DFOEnergyValue);
T = siz(2);
PFOEnergyValue = repelem(0,T);
PFOEnergyValue(1) = DFOEnergyValue(1);
cont = [1];
for t = 2:T
    kh = sum(PFOEnergyValue(1:t-1));
    s = DFOEnergyValue(t);
    prob = prob_sum_improved(data,t,kh,s);
    if prob >= pt && prices.imbalanceprices(t) >= 0
        PFOEnergyValue(t) = DFOEnergyValue(t);
    else
        [t1,t2] = find_thresholds(data,t,kh,pt);
        if s < 0
            PFOEnergyValue(t) = t1;
        else
            PFOEnergyValue(t) = t2;
        end
    end
    if PFOEnergyValue(t)*PFOEnergyValue(t-1) >= 0.01 && t <= T-1
        cont = [cont,t];
    else
        if PFOEnergyValue(t)*PFOEnergyValue(t-1) >= 0.01 && t == T
            cont = [cont,t];
        end
        sc = size(cont);
        if sc(2) > 1
            v = sdpvar(1,sc(2));
            co = [sum(v) == sum(PFOEnergyValue(cont))];
            if PFOEnergyValue(t-1) >= 0.01
                co = [co, 0 <= v <= data.Pmax];
            else
                co = [co, data.L*data.Pmin <= v <= 0];
            end
            optimize(co,prices.spotprices(cont)*v',sett);
            PFOEnergyValue(cont) = value(v);
        end
        cont = [t];
    end
end
PFOcostValue = value(prices.spotprices(1:T) * PFOEnergyValue');
PFOresults = struct();
PFOresults.cost = PFOcostValue;
PFOresults.energy = PFOEnergyValue;
