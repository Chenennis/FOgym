function power = power_cons(data,T_ini,T_fin)
T_out = data.T_out;
pr = prod(size(T_out));
if pr == 1
    power = data.Area * data.c_ht * ((T_fin - T_ini)/(1 - exp(- (data.Area*data.c_ht*data.time)/(data.c*data.m))) + T_ini - data.T_out);
else
    t = data.currenttime;
    power = data.Area * data.c_ht * ((T_fin - T_ini)/(1 - exp(- (data.Area*data.c_ht*data.time)/(data.c*data.m))) + T_ini - T_out(t));
end
end