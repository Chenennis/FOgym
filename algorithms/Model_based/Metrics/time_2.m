function time = time_2(data,T_fin)
T_out = data.T_out;
pr = prod(size(T_out));
if pr == 1
    if (T_fin-data.Tlow)/(data.T_out-data.Tlow+(data.Pmax)/(data.Area*data.c_ht)) < 1 && (T_fin-data.Tlow)/(data.T_out-data.Tlow+(data.Pmax)/(data.Area*data.c_ht)) > 0
        time = -(data.c * data.m)/(data.Area*data.c_ht) * log(1 - (T_fin-data.Tlow)/(data.T_out-data.Tlow+(data.Pmax)/(data.Area*data.c_ht)));
    else
        time = 0;
    end
else
    t_curr = data.currenttime;
    if (T_fin-data.Tlow)/(T_out(t_curr)-data.Tlow+(data.Pmax)/(data.Area*data.c_ht)) < 1 && (T_fin-data.Tlow)/(T_out(t_curr)-data.Tlow+(data.Pmax)/(data.Area*data.c_ht)) > 0
        time = -(data.c * data.m)/(data.Area*data.c_ht) * log(1 - (T_fin-data.Tlow)/(T_out(t_curr)-data.Tlow+(data.Pmax)/(data.Area*data.c_ht)));
    else
        time = 0;
    end
end