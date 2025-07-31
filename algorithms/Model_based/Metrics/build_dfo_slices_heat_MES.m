function slices = build_dfo_slices_heat_MES(data)
 T = data.T;
 slices = [];

 T_in = data.T_in;
 T_min = data.Tlow;
 T_max = data.Thigh;
% first slice: min and max available energy
data.currenttime = 1;
emin = energy_opt(data,T_in,T_min);
emax = energy_opt(data,T_in,T_max);
P1 = Polyhedron([0    emax; 0   emin]);
slices = [slices,P1];
 %build the other slices
 
x1 = emin;
x2 = emin;
x3 = emax;
x4 = emax;

% data.Qmax = data.Qmax*data.percentage/100;
 for t = 1:T-1 
     data.currenttime = t;
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
 end
   
%disp('Done!');

end