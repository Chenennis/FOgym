function slices = build_dfo_slices_heat_simple(data)
 T = data.T;
 period = 3600/data.time;
 slices = [];

 T_in = data.T_in;
 T_min = data.Tlow;
 T_max = data.Thigh;
% first slice: min and max available energy
emin = energy_opt(data,T_in,T_min);
emax = energy_opt(data,T_in,T_min);
P1 = Polyhedron([0    emax; 0   emin]);
slices = [slices,P1];
 %build the other slices

% data.Qmax = data.Qmax*data.percentage/100;
 for t = 1:T-1
     if mod(t+1,period) == 0
         y1 = energy_opt(data,T_min,T_min);
         y2 = energy_opt(data,T_min,T_max);
         y3 = energy_opt(data,T_max,T_min);
         y4 = energy_opt(data,T_max,T_max);
         topleft = [x1,y2];
         bottomleft = [x1,y1];
         bottomcenter = [x3,y3];
         bottomright = [x4,y3];
         topright = [x4,y4];
         topcenter = [x2,y2];
         P1 = Polyhedron([topleft;bottomleft;bottomcenter;bottomright;topright;topcenter]);
         slices = [slices,P1];
         x1 = sum(bottomleft);
         x2 = sum(bottomright);
         x3 = sum(topleft);
         x4 = sum(topright);
     elseif mod(t,period) == 0
         y = energy_opt(data,T_min,T_min);
         vmin = [emin,y];
         vmax = [emax,y];
         P1 = Polyhedron([vmin;vmax]);
         slices = [slices,P1];
     else
 end
   
%disp('Done!');

end
