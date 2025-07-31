function slices = build_dfo_slices_outer_analytic_HP(obj, mCtr, eVars, NumSamples, data)
 T = size(eVars, 2);
 slices = [];
matrixA = exp(-(data.Area*data.c_ht*data.time)/(data.c*data.m));
 
% first, we define some correction values
     addfactor = (1-(matrixA)^2)*data.Pmax;
     correction1 = (data.Thigh - 15 * matrixA) * log(matrixA) * 2 + 3 * (1 - (matrixA)^2) * (data.Pmax - 5);
     correction2 = (1-(matrixA)^2)^2*data.Pmax*(0.5+0.001*(data.Thigh-1));

% first slice: min and max available energy
emin = max(0*matrixA,(data.Tlow-data.T_in)/matrixA);
emax = min(data.Pmax,data.Thigh/matrixA - data.T_in*matrixA - 2*addfactor+correction1);
P1 = Polyhedron([0    emax; 0   emin]);
slices = [slices,P1];
 %build the other slices
 
% data.Thigh = data.Thigh*data.percentage/100;
 for t = 1:T-1

     minright = min(t*data.Pmax,data.Thigh/matrixA - data.T_in*matrixA + (t-3)*addfactor+correction1);
     minleft = min(-t*0*matrixA,data.T_in*matrixA);
     minup = data.Pmax*data.percentage/100;
     mindown = -matrixA*0*data.percentage/100;
     minupright = min((t+1)*data.Pmax,data.Thigh/matrixA - data.T_in*matrixA + (t-2)*addfactor + correction1);
     minupright2 = min((t+1)*data.Pmax,data.Thigh/matrixA - data.T_in*matrixA + (t-2)*addfactor- t*correction2 + correction1);
     mindownleft = min(-(t+1)*0,data.T_in*matrixA);
     A = [1 0;-1 0;0 1;0 -1;1 1;(matrixA)^2 1;-(matrixA)^2 -1;-1 -1];
     b = [minright;minleft;minup;mindown;minupright;minupright2;mindownleft;mindownleft];
     P1 = Polyhedron(A,b);
     slices = [slices,P1];
 end
   
disp('Done!');

end
