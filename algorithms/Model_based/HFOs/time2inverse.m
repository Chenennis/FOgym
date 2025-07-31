function T_fin = time2inverse(data,t_2)
T_fin = data.Tlow + (data.T_out - data.Tlow + data.Pmax/(data.Area*data.c_ht))*(1-exp(-(data.Area*data.c_ht*t_2)/(data.c*data.m)));
end
