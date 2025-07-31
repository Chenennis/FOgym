function slices = build_dfo_slices_TEC_onlyup(obj, mCtr, eVars, NumSamples, data)
 T = size(eVars, 2);
 slices = [];
 Emin = data.Emin;
 Emax = data.Emax;

% first, we define some correction values
     addfactor = (1-(data.L)^2)*data.Pmax;
     correction1 = (data.Qmax - 15 * data.L) * log(data.L) * 2 + 3 * (1 - (data.L)^2) * (data.Pmax - 5);
     correction2 = (1-(data.L)^2)^2*data.Pmax*(0.5+0.001*(data.Qmax-1));

% first slice: min and max available energy
emin = max(0,Emin-(T-1)*data.Pmax);
emax = min(data.Pmax,Emax);
P1 = Polyhedron([0    emax; 0   emin]);
slices = [slices,P1];
 %build the other slices
 
% data.Qmax = data.Qmax*data.percentage/100;
 for t = 1:T-1

     xdownright = min(t*data.Pmax,Emax);
     xupleft = max(0,Emin-(T-t)*data.Pmax);
     minup = data.Pmax;
     mindown = 0;
     xupright = min(xdownright,Emax-data.Pmax);
     yupright = data.Pmax;
     ydownright = max(0,min(Emax-t*data.Pmax,data.Pmax));
     xdownleft = max(0,Emin-(T-t-1)*data.Pmax);
     ydownleft = 0;%max(0,min(Emin-(T-t)*data.Pmax,data.Pmax));
     yupleft = min(data.Pmax,max(0,Emin-(T-t-1)*data.Pmax));
     V = [xdownright,ydownright;xupright,yupright;xupleft,data.Pmax;xupleft,yupleft;xdownleft,ydownleft;xdownright,0];
     P1 = Polyhedron(V);
     slices = [slices,P1];
 end
   
disp('Done!');

end