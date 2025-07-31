function Tfinal = temp_calc(data,T_ini,E)
t_1 = (data.c * data.m)/(data.Area*data.c_ht) * log((T_ini-data.T_out)/(data.Tlow-data.T_out));
t_2 = (E - data.Area*data.c_ht*(data.Tlow-data.T_out)*(data.time-t_1))/(data.Pmax - data.Area*data.c_ht*(data.Tlow-data.T_out));
Tfinal = data.Tlow + (data.T_out - data.Tlow + data.Pmax/(data.Area*data.c_ht))*(1-exp(-(data.Area*data.c_ht*t_2)/(data.c*data.m)));
end
