function T_fin = time2inverse(data,t_2)
T_out = data.T_out;
pr = prod(size(T_out));
if pr == 1
    T_fin = data.Tlow + (data.T_out - data.Tlow + data.Pmax/(data.Area*data.c_ht))*(1-exp(-(data.Area*data.c_ht*t_2)/(data.c*data.m)));
else
    t_curr = data.currenttime;
    T_fin = data.Tlow + (T_out(t_curr) - data.Tlow + data.Pmax/(data.Area*data.c_ht))*(1-exp(-(data.Area*data.c_ht*t_2)/(data.c*data.m)));
end