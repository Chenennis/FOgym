function slices = build_dfo_slices_outer_analytic_uncertainoldpaper(obj, mCtr, eVars, NumSamples, data)
 T = size(eVars, 2);
 slices = [];

% first, we define some correction values
     addfactor = (1-data.pt)*(1-(data.L)^2)*data.Pmax;
     correction1 = (data.Qmax - 15 * data.L) * log(data.L) * 2 + 3 * (1 - (data.L)^2) * (data.Pmax - 5);
     correction2 = (1-(data.L)^2)^2*data.Pmax*(0.5+0.001*(data.Qmax-1));

% first slice: min and max available energy
emin = max(data.Pmin*data.L,(data.Qmin-data.InitialSoC)/data.L);
emax = min(data.Pmax,data.Qmax/data.L - data.InitialSoC*data.L - 2*addfactor+correction1);
P1 = Polyhedron([0    emax; 0   emin]);
slices = [slices,P1];
 %build the other slices
 
% data.Qmax = data.Qmax*data.percentage/100;
 for t = 1:T-1

     minright = min(t*data.Pmax,data.Qmax/data.L - data.InitialSoC*data.L + (t-3)*addfactor+correction1);
     minleft = min(-t*data.Pmin*data.L,data.InitialSoC*data.L);
     minup = data.Pmax*data.percentage/100;
     mindown = -data.L*data.Pmin*data.percentage/100;
     minupright = min((t+1)*data.Pmax,data.Qmax/data.L - data.InitialSoC*data.L + (t-2)*addfactor + correction1);
     minupright2 = min((t+1)*data.Pmax,data.Qmax/data.L - data.InitialSoC*data.L + (t-2)*addfactor- t*correction2 + correction1);
     mindownleft = min(-(t+1)*data.Pmin,data.InitialSoC*data.L);
     A = [1 0;-1 0;0 1;0 -1;1 1;(data.L)^2 1;-(data.L)^2 -1;-1 -1];
     b = [minright;minleft;minup;mindown;minupright;minupright2;mindownleft;mindownleft];
     P1 = Polyhedron(A,b);
     slices = [slices,P1];
 end
   
disp('Done!');

end