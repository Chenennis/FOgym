function time = time_1(data,T_ini)
time = (data.c * data.m)/(data.Area*data.c_ht) * log((T_ini-data.T_out)/(data.Tlow-data.T_out));
end
