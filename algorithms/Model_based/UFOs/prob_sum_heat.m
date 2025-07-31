function p = prob_sum_heat(data,t,k,s)

Q0 = data.InitialSoC;
L = data.L;
Pmin = data.Pmin;
Pmax = data.Pmax;
Qmin = data.Qmin;
Qmax = data.Qmax;
if k > 0
    QQmax = Q0 + L*k;
else
    QQmax = Q0 + k/L;
end
for a1 = 0:t-1
    if a1*Pmax + L*(t-1-a1)*Pmin >= k
        break
    end
end
Qabove = a1*Pmax + L*(t-1-a1)*Pmin;
if Qabove - k >= Pmax
    QQmin = max(Q0 + L*(a1-1)*Pmax + 1/L*(k - (a1-1)*Pmax),Qmin);
else
    QQmin = max(Q0 + 1/L*(t-a1-1)*Pmin + L*(k-(t-a1-1)*Pmin),Qmin);
end
if abs(QQmax - QQmin) < 1e-06
    denom = 1e-16;
else
    denom = QQmax - QQmin;
end
if s-Pmax > 1e-6 || s-Pmin < -1e-6
    p = 0;
elseif s > 0
    p = (Qmax-L*s-QQmin)/(denom);
elseif s < 0
    p = (QQmax-Qmin+s/L+1e-12)/(denom);
else
    p = 2;
end