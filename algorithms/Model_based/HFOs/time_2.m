function time = time_2(data,T_fin)
time = -(data.c * data.m)/(data.Area*data.c_ht) * log(1 - (T_fin-data.Tlow)/(data.T_out-data.Tlow+(data.Pmax)/(data.Area*data.c_ht)));
end
