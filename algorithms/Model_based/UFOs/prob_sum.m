function p = prob_sum(data,t,k,s)

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
    QQmin = max(Q0 + L*k + (1-L)*(t-a2-1)*Pmin,Qmin);
else
    if k >= 0
        QQmin = max(Q0 + a1*(L-1)*Pmax + k,Qmin);
    else
        QQmin = max(Q0 + a1*(L-1)*Pmax + k/L,Qmin);
    end
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