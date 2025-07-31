function [t1,t2] = find_thresholds(data,t,k,pt)

Q0 = data.InitialSoC;
L = data.L;
Pmin = data.Pmin;
Pmax = data.Pmax;
Qmin = data.Qmin;
Qmax = data.Qmax;
if k >= 0
    QQmax = min(Q0 + L*k,Qmax);
else
    QQmax = min(Q0 + k/L,Qmax);
end
for a1 = 0:t-1
    if Q0 + a1*Pmax + L*(t-1-a1)*Pmin >= k
        break
    end
end
for a2 = 0:t-2
    if Q0 + a2*Pmax + L*(t-2-a2)*Pmin >= k
        break
    end
end
if a1 == a2
    QQmin = max(Q0 + L*k + (1/L-L)*(t-a2-1)*Pmin,Qmin);
else
    QQmin = max(Q0 + a1*(L-1)*Pmax + k,Qmin);
end

t2 = min((Qmax - QQmin - pt*(QQmax-QQmin))/L,Pmax);
t1 = max(L*(Qmin - QQmax + pt*(QQmax-QQmin)),L*Pmin);
end