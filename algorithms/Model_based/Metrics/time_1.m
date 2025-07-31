function time = time_1(data,T_ini)
T_out = data.T_out;
pr = prod(size(T_out));
if pr == 1
    if (T_ini-data.T_out)/(data.Tlow-data.T_out) > 1
        time = (data.c * data.m)/(data.Area*data.c_ht) * log((T_ini-data.T_out)/(data.Tlow-data.T_out));
    else
        time = 0;
    end
else
    t_curr = data.currenttime;
    if (T_ini-T_out(t_curr))/(data.Tlow-T_out(t_curr)) > 1
        time = (data.c * data.m)/(data.Area*data.c_ht) * log((T_ini-T_out(t_curr))/(data.Tlow-T_out(t_curr)));
    else
        time = 0;
    end
end