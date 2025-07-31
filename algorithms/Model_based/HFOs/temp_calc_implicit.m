function Tfinal = temp_calc_implicit(data,T_ini,E)
t_1 = time_1(data,T_ini);
t_2 = (data.time*E - data.Area*data.c_ht*(data.Tlow-data.T_out)*(data.time-t_1))/(data.Pmax - data.Area*data.c_ht*(data.Tlow-data.T_out));
Tfinal = time2inverse(data,t_2);
end
