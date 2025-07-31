function power = power_cons(data,T_ini,T_fin)
power = data.Area * data.c_ht * ((T_fin - T_ini)/(1 - exp(- (data.Area*data.c_ht*data.time)/(data.c*data.m))) + T_ini - data.T_out);
end
