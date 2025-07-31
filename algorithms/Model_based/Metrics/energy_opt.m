function energy = energy_opt(data,T_ini,T_fin)
T_out = data.T_out;
pr = prod(size(T_out));
if pr == 1
    t_1 = time_1(data,T_ini);
    t_2 = time_2(data,T_fin);
    if ~isreal(t_1) || t_1 > data.time || t_1 < 0
        t_1 = 0;
    end
    if ~isreal(t_2) || t_2 > data.time || t_2 < 0
        t_2 = 0;
    end
    energy = data.Area * data.c_ht * (data.Tlow-data.T_out) * (data.time - t_1 - t_2) + data.Pmax * t_2;
    energy = energy/data.time;
else
    t_curr = data.currenttime;
    t_1 = time_1(data,T_ini);
    t_2 = time_2(data,T_fin);
    if ~isreal(t_1) || t_1 > data.time || t_1 < 0
        t_1 = 0;
    end
    if ~isreal(t_2) || t_2 > data.time || t_2 < 0
        t_2 = 0;
    end
    energy = data.Area * data.c_ht * (data.Tlow-T_out(t_curr)) * (data.time - t_1 - t_2) + data.Pmax * t_2;
    energy = energy/data.time;
end