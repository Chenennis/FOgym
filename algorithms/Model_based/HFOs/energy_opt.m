function energy = energy_opt(data,T_ini,T_fin)
t_1 = time_1(data,T_ini);
t_2 = time_2(data,T_fin);
energy = data.Area * data.c_ht * (data.Tlow-data.T_out) * (data.time - t_1 - t_2) + data.Pmax * t_2;
energy = energy/data.time;
end
