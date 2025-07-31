format shortG
MRes = repelem(0,2,9);
possibilities = [0,0.1,0.2,0.3,0.4,0.55,2];
for time = 24:24
for poss = 7:7
%for TType = 1:2
parallel = 1;

counter = 1;
Length = 124;
RMatrix = repelem(0,7,Length);
QStatus = repelem(0,Length+1);
QStatus(1) = 0;
daysahead = 0;
      
%UType = TType - 1;
Q0 = 0;
eta = 0.948;
Qmin = 0;
Qmax = 14;
Pmin = -5;
Pmax = 5;
T = time;
%decay = 0.99;
decay = 1;
Cost = 0;
U = T;
%Type = (-1)^UType * UType;
Type = possibilities(poss);%2*UType;
spos = T*(counter - 1)+ T*Length*(parallel-1);
options = sdpsettings('verbose',0);
constraints = [];
constraints2 = [];

up = sdpvar(1,T);
un = sdpvar(1,T);
Q = sdpvar(1,T+1);
Q2 = sdpvar(1,T+1);
psum = sdpvar(1,T+1);
%psum = sdpvar(1);

Q(1) = Q0;
VCost = 0;

VPmin = repelem(eta*Pmin,T);
VPmax = repelem(Pmax,T);
VQmin = repelem(Qmin,T+1);
VQmax = repelem(Qmax,T+1);
production = repelem(0,T);
consumption = repelem(0,T);

%M = readmatrix('C:\Users\Fab\Documents\Python\DK1data.xls','Sheet','Matrix','Range','A1:A14200');
M = readmatrix('C:\Users\Fab\Documents\Postdoc\Datasets\Spot Prices.xls','Sheet','Foglio1','Range','A1:A25896');
N = readmatrix('C:\Users\Fab\Documents\Postdoc\Datasets\Regulation Prices.xls','Sheet','Foglio1','Range','A1:A25896');
% M = readmatrix('/home/ubuntu/Spot Prices.xls','Sheet','Foglio1','Range','A1:A25896');
% N = readmatrix('/home/ubuntu/Regulation Prices.xls','Sheet','Foglio1','Range','A1:A25896');
tariffbuy = M(1+spos+daysahead:T+spos+daysahead,1:1)';
penalty = N(1+spos+daysahead:T+spos+daysahead,1:1)';
net = repelem(0,T);

production1 = [0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0.000352611	0.002705207	0.007085441	0.015782473	0.029141966	0.048603202	0.075536986	0.11805863	0.175308317	0.249271892	0.337440498	0.432768055	0.516657487	0.594715107	0.70758545	0.822326545	0.935458515	1.050358094	1.146458791	1.255553655	1.364499172	1.449006351	1.534798565	1.607588482	1.691386061	1.763761529	1.78915626	1.805596397	1.833910126	1.857642302	1.93907999	1.970953983	1.965218818	1.946961013	1.926062582	1.899741229	1.8854383	1.845752521	1.786141276	1.708643665	1.610091085	1.53281859	1.479163456	1.359065759	1.269951251	1.127126059	0.999416366	0.86240836	0.735646688	0.665027329	0.524227031	0.423221386	0.368801033	0.2987322	0.233434252	0.174488625	0.122623194	0.079999172	0.050105466	0.033720392	0.020305175	1.00E-02	0.004105127	0.001053249	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0];
%consumption1 = [0.68306805	0.60054225	0.58247175	0.5644002	0.48517245	0.40594365	0.39169515	0.3774456	0.3803919	0.38333715	0.3873009	0.39126465	0.39344865	0.3956316	0.38466435	0.37369605	0.34382985	0.31396365	0.34350435	0.37304505	0.39852645	0.4240068	0.4359453	0.4478838	0.45915345	0.4704231	0.62916315	0.7879032	0.85912785	0.93035145	0.991032	1.0517115	1.07035845	1.0890054	1.03393185	0.9788583	0.9218097	0.8647611	0.8049867	0.74521125	0.79014915	0.835086	0.80749095	0.7798959	0.8103753	0.84085365	0.74139555	0.64193745	0.75636855	0.87079965	0.81702705	0.76325445	0.86672775	0.97020105	0.9279102	0.8856183	0.80878455	0.7319508	0.79030455	0.8486583	0.80317965	0.757701	0.8749587	0.9922164	1.01143035	1.0306443	1.0872288	1.1438133	1.2178803	1.29194625	1.48141035	1.6708734	1.55056545	1.4302575	1.51807215	1.6058868	1.554252	1.50261615	1.4446551	1.38669405	1.46159475	1.5364944	1.4443401	1.35218475	1.434384	1.5165822	1.4396487	1.36271415	1.26438795	1.16606175	1.0685115	0.9709602	0.85437345	0.73778565	0.75168975	0.76559385];
% for k = 1:T
%     production(k) = (production1(4*k) + production1(4*k-1) + production1(4*k-2) + production1(4*k-3))/4;
%     consumption(k) = (consumption1(4*k) + consumption1(4*k-1) + consumption1(4*k-2) + consumption1(4*k-3))/4;
% end

%tariffsell = repelem(0.0491, T);
%net1 = repelem(0, 96);
%tariffbuy = [0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.0508	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627	0.1627];
%tariffbuy = [20.72 17.96 17.09 12.44 12.48 13.74 20.20 24.90 26.01 23.23 21.07 20.50 19.45 15.20 16.19 16.14 17.40 20.06 23.53 27.51 26.22 24.70 25.46 20.78];
%tariffbuy = [21.04 17.21 14.93 12.63 10.92 10.96 10.88 14.04 15.36 16.00 15.70 16.06 16.94 13.37 10.76 10.64 13.64 17.00 21.06 25.36 28.56 29.14 32.33 28.91];
%net = consumption - production;



if Type == 2.1
    for k = 1:T
   Q(k+1) = decay*Q(k) + eta * max(psum(k+1),0) + 1/eta * min(psum(k),0);
%   VCost = VCost + Cost*(up(k)+un(k));
    end
    constraints = [constraints, repelem(eta*Pmin,T+1) <= psum <= repelem(Pmax,T+1), VQmin <= Q <= VQmax];
    net1 = net + psum(2:T+1);
end

% Model is taken from:
%       https://control.me.berkeley.edu/~sanandaji/my_papers/Allerton2013_TCL.pdf
% LTI Representation
%    x(t+Ts) = A*x(t) + B*u(t)+f 
%       y(t) = C*x(t) + D*u(t) 
%
if Type ~= 2.1
      Qflexmax = (Qmax - Q(1))/eta;
    Qflexmin = eta*(Qmin - Q(1));

if Type == 0.3
      for k = 1:T
          Q(k+1) = decay*Q(k) + eta *max(0,psum(k+1)) + 1/eta * min(0,psum(k+1));
%        constraints = [constraints,Qmin <= Q(k+1) <= Qmax ]
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
    end
    Qflexmin2 = max(Qflexmin, eta*Pmin*T);
    Qflexmax2 = min(Qflexmax, Pmax*T/eta);
    
    if -Qflexmin2 >= Qflexmax2
        for k = 1:T
            constraints = [constraints,eta*Pmin<= psum(k+1) <= Qflexmax2/T];
        end
        constraints = [constraints,sum(psum(2:k+1)) >= Qflexmin2+Qflexmax2];
    else
            for k = 1:T
            constraints = [constraints,Qflexmin2/T <= psum(k+1) <= Pmax];
        end
        constraints = [constraints,sum(psum(2:k+1)) <= Qflexmin2+Qflexmax2];
    end

%end
%constraints = [constraints, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax];%, norm(un.*up) <= 0.01];
% constraints = [constraints ,min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax, minsum <= sum(up)+sum(un) <= Qflexmax ];%, un <= 0, 0 <= up,  norm(un.*up) <= 0.01]
net1 = net + psum(2:T+1);
 
elseif Type == 0.4
if Q(1) >= T/2*Pmax*(1 - eta) + Qmax - Pmax
        maxsum = T/2*Pmax*(1 - eta^2) + eta* (Qmax-Q(1));
    elseif Q(1) >= (T/2*(1 - eta) - eta)*Pmax + Qmax - Pmax
        maxsum = (T/2-1)*(1/eta - eta)*Pmax + (Qmax-Q(1))/eta;
    elseif Q(1) >= (T/2*(1 - eta) - 2*eta - 1)*Pmax + Qmax
        maxsum = (T/2+1)*Pmax*(1 - eta^2) + eta* (Qmax-Q(1));
    else
        maxsum = (T/2-2)*(1/eta - eta)*Pmax + (Qmax-Q(1))/eta;
end
    for k = 1:T
        Q(k+1) = decay*Q(k) + eta *max(0,psum(k+1)) + 1/eta * min(0,psum(k+1));
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
 end


%end
%constraints = [constraints, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax];%, norm(un.*up) <= 0.01];
 constraints = [constraints, Qflexmin <= sum(psum(2:T+1)) <= maxsum, repelem(eta*Pmin,T+1) <= psum <= repelem(Pmax,T+1)];%, VQmin <= Q <= VQmax];%, un <= 0, 0 <= up,  norm(un.*up) <= 0.01]
net1 = net + psum(2:T+1);

elseif Type == 0.7
    up = sdpvar(1,T+1);
    un = sdpvar(1,T+1);
    
    
    constraints = [up(1) == 0; un(1) == 0];
        for k = 1:T
        SoC = Q(1) + eta*sum(up(1:k)) + 1/eta * sum(un(1:k));
        constraints = [constraints, 0 <= up(k+1) <= min(Pmax,(Qmax-SoC)/eta), max(Pmin,eta*(Qmin-SoC)) <= un(k+1) <= 0]; 
        Q(k+1) = decay*Q(k) + eta * up(k+1) + 1/eta * un(k+1);
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
        end
    


%end
%constraints = [constraints, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax];%, norm(un.*up) <= 0.01];
% constraints = [constraints, Qflexmin <= sum(psum(2:T+1)) <= maxsum, repelem(eta*Pmin,T+1) <= psum <= repelem(Pmax,T+1)]% VQmin <= Q <= VQmax];%, un <= 0, 0 <= up,  norm(un.*up) <= 0.01]
net1 = net + up(2:k+1) + un(2:k+1);

    elseif Type == 0.1
        for k = 1:T
        Q(k+1) = decay*Q(k) + eta *max(0,psum(k+1)) + 1/eta * min(0,psum(k+1));
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
 end


%end
%constraints = [constraints, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax];%, norm(un.*up) <= 0.01];
% constraints = [constraints, VQmin <= Q <= VQmax];%, un <= 0, 0 <= up,  norm(un.*up) <= 0.01]
    for k = 1:T+1
        mmin = min(max(eta*Pmin,-Qflexmin-(k-1)*eta*Pmin),0);
        mmax = max(min(Pmax,Qflexmax-(k-1)*Pmax),0);
    constraints = [constraints, mmin <= psum(k) <= mmax ];
    end
 %   constraints = [constraints, repelem(eta*Pmin,T+1) <= psum <= repelem(Pmax,T+1)]
net1 = net + psum(2:T+1);

elseif Type == 0.2
     for k = 1:T
        Q(k+1) = decay*Q(k) + eta *max(0,psum(k+1)) + 1/eta * min(0,psum(k+1));
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
 end


%end
%constraints = [constraints, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax];%, norm(un.*up) <= 0.01];
 %constraints = [constraints, VQmin <= Q <= VQmax];%, un <= 0, 0 <= up,  norm(un.*up) <= 0.01]
 %  for k = 1:T+1
 %  constraints = [constraints,Qflexmin/T <= psum(k) <= Qflexmax/T];
 %  end
    constraints = [constraints, repelem(Qflexmin/T,T+1) <= psum <= repelem(Qflexmax/T,T+1)];
net1 = net + psum(2:T+1);

elseif Type == 0.55
     for k = 1:T
        Q(k+1) = decay*Q(k) + eta *max(0,psum(k+1)) + 1/eta * min(0,psum(k+1));
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
 end


%end
%constraints = [constraints, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax];%, norm(un.*up) <= 0.01];
 %constraints = [constraints, VQmin <= Q <= VQmax];%, un <= 0, 0 <= up,  norm(un.*up) <= 0.01]
    constraints = [constraints, repelem(eta*Pmin,T+1) <= psum <= repelem(Pmax,T+1)];
net1 = net + psum(2:T+1);
    else
    
% if Type == -1
% A = [decay];
% B1 = [eta];
% B2 = [1/eta];
% f = [0];
% C = [0];
% D = [1];
% lti = LTISystem('A', A, 'B', B1, 'C', C, 'D', D);
% % local1 = LTISystem('A', A, 'B', B1, 'C', C, 'D', D);
% % local2 = LTISystem('A', A, 'B', B2, 'C', C, 'D', D);
% % P1 = Polyhedron('lb',0);
% % P2 = Polyhedron('ub',0);
% % % P1 = Polyhedron('A',[-1;1],'b',[0;Pmax]);
% % % P2 = Polyhedron('A',[-1;1],'b',[Pmin;0]);
% % % Q1 = Polyhedron('A',[-1;1],'b',[0;Qmax]);
% % local1.setDomain('u',P1);
% % local2.setDomain('u',P2);
% % local1.setDomain('x',Q1);
% % local2.setDomain('x',Q1);
% % lti = PWASystem([local1, local2]);
% lti.u.min = Pmin;
% lti.u.max = Pmax;
% lti.x.min = Qmin;
% lti.x.max = Qmax;
% lti.initialize(Q0);
% 
%  % Outer approximation performance
% genTimeO = [];
% for T1 = 1%:4:24
%  lti.instantiate(U);
%  %lti.instantiate(1);
%  %sdpsettings('solver','SCIP');
%  %yalmip('solver','scip');
%  F1 = FlexSystem(lti);
% end
% end
%if Type == 0
    A = [decay];
B3 = [eta,1/eta]; %FOR DFOS ONLY
f = [0];
C = [0];
D1 = [1,1]; %FOR DFOS ONLY
PminV = [0;eta*Pmin]; %FOR DFOS ONLY
PmaxV = [Pmax;0]; %FOR DFOS ONLY
lti = LTISystem('A', A, 'B', B3, 'C', C, 'D', D1); %FOR DFOS ONLY
lti.u.min = PminV;
lti.u.max = PmaxV;
lti.x.min = Qmin;
lti.x.max = Qmax;
lti.initialize(Q0);

 % Outer approximation performance
genTimeO = [];
for T1 = 1%:4:24
 lti.instantiate(U);
 %lti.instantiate(1);
 F1 = FlexSystem(lti);
 
end
%end
  Type1 = Type;
  if Type == 0.1
      Type1 = 0.5;
  end
  if Type == 0.2
      Type1 = 0.5;
  end
  if Type == 0.3
      Type1 = 0.5;
  end
  if Type == 0.4
      Type1 = 0.5;
  end
 tic
 ODFO = DFOSystem(F1,Type);
 tGen = toc;
 
 genTimeO = [genTimeO; T1, tGen];


%ODFO.plot_slices()
% if Type == -1
% [VPmin,VPmax] = computeEnergyLUBounds(ODFO);
% VPmin = VPmin';
% VPmax = VPmax';
% 
% for k = 1:U-1
%    %Q(k+1) = decay*Q(k) + eta*max([u(k) 0]) + 1/eta * min([u(k) 0]);
%    Q(k+1) = decay*Q(k) + eta *up(k) + 1/eta * un(k);
%   % VCost = VCost + Cost*up((TT-1)*U+k);
% end
% end
% if Type == 0
 for k = 1:U
    P1 = ODFO.slices(k);
    Vec = size(P1.H);
    for j = 1:Vec(1)
        vt1 = repelem(0,T+1);
        for yetanother = 1:k
        vt1(yetanother) = P1.H(j,1);
        end
        vt1(k+1) = P1.H(j,2);
        constraints = [constraints, vt1*(psum)' <= P1.H(j,3)];%, repelem(eta*Pmin,T+1) <= psum <= repelem(Pmax,T+1)];
    end
%     P2 = Polyhedron([1,0;-1,0;0,1;0,-1],[value(psum);-value(psum);Pmax;-Pmin]);
%     P3 = and(P1,P2);
%     VPmin(k) = P3.V(2,2);
%     VPmax(k) = P3.V(1,2);
%     disp(value(VPmin((TT-1)*U+1+k)));
%     disp(value(VPmax((TT-1)*U+1+k)));
%     disp('check');
   %Q(k+1) = decay*Q(k) + eta*max([u(k) 0]) + 1/eta * min([u(k) 0]);
 %  Q(k+1) = decay*Q(k) + eta *up(k) + 1/eta * un(k);
 Q(k+1) = decay*Q(k) + eta *max(0,psum(k+1)) + 1/eta * min(0,psum(k+1));
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
 end


%end
%constraints = [constraints, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax];%, norm(un.*up) <= 0.01];
% constraints = [constraints, VQmin <= Q <= VQmax];%, un <= 0, 0 <= up,  norm(un.*up) <= 0.01]
net1 = net + psum(2:T+1);
end
end
ccost = tariffbuy(1:T) * net1' + VCost;

timer = 0;
for k = 1:100
tic
Solution = optimize(constraints, ccost,options);
timer = timer + toc;
%if mod(k,500) == 0
%    disp(possibilities(poss));
%    disp(k);
%end
end
topt = timer/100;
MRes(1,poss) = possibilities(poss);
MRes(2,poss) = topt;

disp(value(time));
disp(value(MRes));
end
end