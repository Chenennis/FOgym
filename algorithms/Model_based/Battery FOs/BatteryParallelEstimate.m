format shortG
%for TType = 1:2
parfor parallel = 1:12
%parallel = 1;
Length = 62;
RMatrix = repelem(0,3,Length);
QStatus = repelem(0,Length+1);
QStatus(1) = 0;
daysahead = 0;
    for counter = 1:Length
      
%UType = TType - 1;
Q0 = QStatus(counter);
eta = 0.948;
Qmin = 0;
Qmax = 14;
Pmin = -5; 
Pmax = 5;
T = 12;
%decay = 0.99;
decay = 1;
Cost = 0;
U = T;
Tmezzi = T/2;
Umezzi = U/2;
%Type = (-1)^UType * UType;
Type = 0.55;%2*UType;
valid = 0;
spos = T*(counter - 1)+ T*Length*(parallel-1);
%options = sdpsettings('solver','cplex');
constraints = [];

up = sdpvar(1,T);
un = sdpvar(1,T);
Q = sdpvar(1,T+1);
Q1 = sdpvar(1,T+1);
psum1 = sdpvar(1,Tmezzi+1);
psum2 = sdpvar(1,Tmezzi+1);
psum = sdpvar(1,T+1);

Q(1) = Q0;
Q1(1) = Q0;
VCost = 0;

VPmin = repelem(eta*Pmin,T);
VPmax = repelem(Pmax,T);
VQmin = repelem(Qmin,T+1);
VQmax = repelem(Qmax,T+1);
production = repelem(0,T);
consumption = repelem(0,T);

%M = readmatrix('C:\Users\Fab\Documents\Python\DK1data.xls','Sheet','Matrix','Range','A1:A14200');
% M = readmatrix('C:\Users\Fab\Documents\Postdoc\Datasets\Spot Prices.xls','Sheet','Foglio1','Range','A1:A25896');
% N = readmatrix('C:\Users\Fab\Documents\Postdoc\Datasets\Regulation Prices.xls','Sheet','Foglio1','Range','A1:A25896');
M = readmatrix('/home/ubuntu/Spot Prices.xls','Sheet','Foglio1','Range','A1:A25896');
N = readmatrix('/home/ubuntu/Regulation Prices.xls','Sheet','Foglio1','Range','A1:A25896');
tariffbuy = M(1+spos+daysahead:T+spos+daysahead,1:1)';
penalty = N(1+spos+daysahead:T+spos+daysahead,1:1)';
net = repelem(0,T);

% for k = 1:T
%     production(k) = (production1(4*k) + production1(4*k-1) + production1(4*k-2) + production1(4*k-3))/4;
%     consumption(k) = (consumption1(4*k) + consumption1(4*k-1) + consumption1(4*k-2) + consumption1(4*k-3))/4;
% end

%tariffsell = repelem(0.0491, T);
%net1 = repelem(0, 96);
%net = consumption - production;

Tmezzi = T/2;
Umezzi = U/2;

%%%%% Instantiate an exact model
A = [decay];
B3 = [1/eta]; %FOR DFOS ONLY
f = [0];
C = [0];
D1 = [1]; %FOR DFOS ONLY
PminV = [eta*Pmin]; %FOR DFOS ONLY
PmaxV = [0]; %FOR DFOS ONLY
lti = LTISystem('A', A, 'B', B3, 'C', C, 'D', D1); %FOR DFOS ONLY
lti.u.min = PminV;
lti.u.max = PmaxV;
lti.x.min = Qmin;
lti.x.max = Qmax;
lti.initialize(Q0);

P = Polyhedron('lb', [Q0], 'ub', [Q0]);
lti.x.with('initialSet');
lti.x.initialSet = P;

 % Outer approximation performance
genTimeO = [];
lti.instantiate(Umezzi);
F1 = FlexSystem(lti);

% Generate approximations




if Type == 2.1
    for k = 1:Tmezzi
   Q(k+1) = decay*Q(k) + eta *up(k);
   VCost = VCost + Cost*(up(k)+un(k));
   constraints = [constraints, un(k) == 0];
    end
    for k = Tmezzi+1:T
           Q(k+1) = decay*Q(k) + 1/eta * un(k);
   VCost = VCost + Cost*(up(k)+un(k));
   constraints = [constraints, up(k) == 0];
    end
    %constraints = [constraints, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax];
    constraints = [constraints, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax];
    net1 = net + up + un;
    ccost = tariffbuy * net1' + VCost;
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

if or(Type == 0.3, Type == 0.4) == 1
  
%     if Q(1) <= T/2*Pmax*(1 - eta)
%         minsum = T/2*Pmax*(1 - eta^2) - eta* Q(1);
%     elseif Q(1) <= (T/2*(1 - eta)+ eta)*Pmax
%         minsum = T/2*(1/eta - eta)*Pmax - Q(1)/eta;
%     elseif Q(1) <= (T/2*(1 - eta)+ eta + 1)*Pmax
%         minsum = (T/2-1)*Pmax*(1 - eta^2) - eta* Q(1);
%     else
%         minsum = (T/2+1)*(1/eta - eta)*Pmax - Q(1)/eta;
%     end
    for k = 1:T
          Q(k+1) = decay*Q(k) + eta *max(0,psum(k+1)) + 1/eta * min(0,psum(k+1));
%        constraints = [constraints,Qmin <= Q(k+1) <= Qmax ]
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
    end
    Qflexmin2 = max(Qflexmin, eta*Pmin*T);
    Qflexmax2 = min(Qflexmax, Pmax*T/eta);
    
        for k = 1:Tmezzi
            constraints = [constraints,0<= psum(k+1) <= Pmax];
        end
        constraints = [constraints,sum(psum(2:Tmezzi+1)) <= Qflexmax];
            for k = Tmezzi+1:T
            constraints = [constraints,eta*Pmin <= psum(k+1) <= 0];
        end
        constraints = [constraints,sum(psum(Tmezzi+2:T+1)) >= -eta*(Q(1)+eta*sum(psum(2:Tmezzi+1)))];
   % end

%end
%constraints = [constraints, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax];%, norm(un.*up) <= 0.01];
% constraints = [constraints ,min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax, minsum <= sum(up)+sum(un) <= Qflexmax ];%, un <= 0, 0 <= up,  norm(un.*up) <= 0.01]
net1 = net + psum(2:T+1);
 ccost = tariffbuy * psum(2:T+1)' + VCost;
elseif Type == 0.4
minsum = -eta*Qmax;
    maxsum = (Qmax-Q(1))/eta;
   
    for k = 1:Tmezzi
        Q(k+1) = decay*Q(k) + eta * psum(k+1);
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
    end
 for k = Tmezzi+1: T
     Q(k+1) = decay*Q(k) + 1/eta * psum(k+1);
 end


%end
%constraints = [constraints, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax];%, norm(un.*up) <= 0.01];
 constraints = [constraints, 0 <= sum(psum(2:Tmezzi+1)) <= maxsum, minsum <= sum(psum(Tmezzi+2:T+1)) <= 0, 0 <= psum(2,Tmezzi+1) <= repelem(Pmax,T+1), repelem(eta*Pmin,T+1) <= psum(Tmezzi+2,T+1) <= 0];% , VQmin <= Q <= VQmax];
net1 = net + psum(2:T+1);
ccost = tariffbuy * net1' + VCost;
    elseif Type == 0.1
        for k = 1:Tmezzi
        Q(k+1) = decay*Q(k) + eta *psum(k+1);
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
 end
        for k = Tmezzi+1:T
        Q(k+1) = decay*Q(k) + 1/eta * psum(k+1);
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
 end


%end
%constraints = [constraints, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax];%, norm(un.*up) <= 0.01];
 %constraints = [constraints, VQmin <= Q <= VQmax];%, un <= 0, 0 <= up,  norm(un.*up) <= 0.01]
    for k = 1:Tmezzi+1
    constraints = [constraints, 0 <= psum(k) <= max(min(Pmax,Qflexmax-(k-1)*Pmax),0)];
    end
    for k = Tmezzi+2:T+1
    constraints = [constraints,min(max(eta*Pmin,-eta*Qmax-(k-1)*eta*Pmin),0) <= psum(k) <= 0];
    end
    constraints = [constraints, repelem(eta*Pmin,T+1) <= psum <= repelem(Pmax,T+1)];%, Qmin <= Q <= Qmax];
net1 = net + psum(2:T+1);
ccost = tariffbuy * net1' + VCost;
elseif Type == 0.2
     for k = 1:Tmezzi
        Q(k+1) = decay*Q(k) + eta *psum(k+1);
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
 end
        for k = Tmezzi+1:T
        Q(k+1) = decay*Q(k) + 1/eta * psum(k+1);
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
 end

%end
%constraints = [constraints, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax];%, norm(un.*up) <= 0.01];
 %constraints = [constraints, VQmin <= Q <= VQmax];%, un <= 0, 0 <= up,  norm(un.*up) <= 0.01]
   for k = 1:Tmezzi+1
   constraints = [constraints,0 <= psum(k) <= 2*Qflexmax/T];
   end
   for k = Tmezzi+2:T+1
   constraints = [constraints, 2*eta*(-eta*sum(psum(2:Tmezzi+1))-Q(1))/T <= psum(k) <= 0 ];
   end
    constraints = [constraints, repelem(eta*Pmin,T+1) <= psum <= repelem(Pmax,T+1)];
net1 = net + psum(2:T+1);
ccost = tariffbuy * net1' + VCost;
elseif Type == 0.55
      for k = 1:Tmezzi
        Q(k+1) = decay*Q(k) + eta *psum(k+1);
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
      end
      Qref = Q(Tmezzi+1);
        for k = Tmezzi+1:T
        Q(k+1) = decay*Q(k) + 1/eta * psum(k+1);
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
 end


%end
%constraints = [constraints, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q <= VQmax];%, norm(un.*up) <= 0.01];
 %constraints = [constraints, VQmin <= Q <= VQmax];%, un <= 0, 0 <= up,  norm(un.*up) <= 0.01]
    constraints = [constraints, repelem(0,Tmezzi+1) <= psum(1:Tmezzi+1) <= repelem(min(Pmax,Qmax-Q0),Tmezzi+1), repelem(eta*max(Pmin,Qmin-Qref),Tmezzi) <= psum(Tmezzi+2:T+1) <= repelem(0,Tmezzi)];
net1 = net + psum(2:T+1);
ccost = tariffbuy * net1' + VCost;
else
        constraints111 = [];
        psum3 = sdpvar(1,T+1);
 psum11 = sdpvar(1,Tmezzi+1);
 psum22 = sdpvar(1,Tmezzi+1);
%for estimate = 1:6
    net1 = net(1:Tmezzi);
    net2 = net(Tmezzi+1:T);
    tariffbuy1 = tariffbuy(1:Tmezzi);
    tariffbuy2 = tariffbuy(Tmezzi+1:T);
    A = [decay];
B3 = [eta]; %FOR DFOS ONLY
f = [0];
C = [0];
D1 = [1]; %FOR DFOS ONLY
PminV = [0]; %FOR DFOS ONLY
PmaxV = [Pmax]; %FOR DFOS ONLY
lti = LTISystem('A', A, 'B', B3, 'C', C, 'D', D1); %FOR DFOS ONLY
lti.u.min = PminV;
lti.u.max = PmaxV;
lti.x.min = Qmin;
lti.x.max = Qmax;
lti.initialize(Q0);

 % Outer approximation performance
genTimeO = [];
for T1 = 1%:4:24
 lti.instantiate(Umezzi);
 %lti.instantiate(1);
 F1 = FlexSystem(lti);
 
end
%end
  
 tic
 ODFO1 = DFOSystem(F1,Type);
 tGen = toc;
 
 genTimeO = [genTimeO; T1, tGen];



    A = [decay];
B3 = [1/eta]; %FOR DFOS ONLY
f = [0];
C = [0];
D1 = [1]; %FOR DFOS ONLY
PminV = [eta*Pmin]; %FOR DFOS ONLY
PmaxV = [0]; %FOR DFOS ONLY
lti = LTISystem('A', A, 'B', B3, 'C', C, 'D', D1); %FOR DFOS ONLY
lti.u.min = PminV;
lti.u.max = PmaxV;
lti.x.min = Qmin;
lti.x.max = Qmax;
lti.initialize(Qmax);

 % Outer approximation performance
genTimeO = [];
for T1 = 1%:4:24
 lti.instantiate(Umezzi);
 %lti.instantiate(1);
 F1 = FlexSystem(lti);
 
end
%end
  
 tic
 ODFO2 = DFOSystem(F1,Type);
 tGen = toc;
 
 genTimeO = [genTimeO; T1, tGen];


%ODFO.plot_slices()



 for k = 1:Tmezzi
    P1 = ODFO1.slices(k);
    Vec = size(P1.H);
    for j = 1:Vec(1)
        vt1 = repelem(0,Tmezzi+1);
        for yetanother = 1:k
        vt1(yetanother) = P1.H(j,1);
        end
        vt1(k+1) = P1.H(j,2);
        constraints111 = [constraints111, vt1*(psum11)' <= P1.H(j,3)];
    end
 Q1(k+1) = decay*Q1(k) + eta *max(0,psum11(k+1));
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
 end 
 for k = 1:Tmezzi
    P1 = ODFO2.slices(k);
    Vec = size(P1.H);
    for j = 1:Vec(1)
        vt1 = repelem(0,Tmezzi+1);
        for yetanother = 1:k
        vt1(yetanother) = P1.H(j,1);
        end
        vt1(k+1) = P1.H(j,2);
        constraints111 = [constraints111, vt1*(psum22)' <= P1.H(j,3)];
    end
 Q1(Tmezzi+k+1) = decay*Q1(Tmezzi+k) + 1/eta * min(0,psum22(k+1));
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
 constraints111 = [constraints111, Q0 + eta*sum(psum11(2:Tmezzi+1)) >= -(sum(psum22(2:Tmezzi+1)))/eta+10e-2];
 end
 %constraints111 = [constraints111, VQmin <= Q <= VQmax]
 constraints111 = [constraints111, repelem(0,Tmezzi+1) <= psum11 <= repelem(Pmax,Tmezzi+1), repelem(eta*Pmin,Tmezzi+1) <= psum22 <= repelem(0,Tmezzi+1)];
 net11 = net1 + psum11(2:Tmezzi+1);
net22 = net2 + psum22(2:Tmezzi+1);
ccost111 = tariffbuy1(1:Tmezzi) * net11' + tariffbuy2(1:Tmezzi) * net22' + VCost;

Solution1 = optimize(constraints111,ccost111);
%for k =1:Tmezzi+1
%    if isnan(value(Q(k))) == 1
%        disp('NAN FOUND')
%    end
%end

%CUT FROM HERE
Qmiddle = value(Q1(Tmezzi+1));
if Qmiddle >= 10e-2
    
constraints = [constraints, Qmiddle <= Q0 + eta*sum(psum1(2:Tmezzi+1))];

  A = [decay];
B3 = [1/eta]; %FOR DFOS ONLY
f = [0];
C = [0];
D1 = [1]; %FOR DFOS ONLY
PminV = [eta*Pmin]; %FOR DFOS ONLY
PmaxV = [0]; %FOR DFOS ONLY
lti = LTISystem('A', A, 'B', B3, 'C', C, 'D', D1); %FOR DFOS ONLY
lti.u.min = PminV;
lti.u.max = PmaxV;
lti.x.min = Qmin;
lti.x.max = Qmax;
lti.initialize(Qmiddle);

 % Outer approximation performance
genTimeO = [];
for T1 = 1%:4:24
 lti.instantiate(Umezzi);
 %lti.instantiate(1);
 F1 = FlexSystem(lti);
 
end
%end
  
 tic
 ODFO2 = DFOSystem(F1,Type);
 tGen = toc;
 
 genTimeO = [genTimeO; T1, tGen];


%ODFO.plot_slices()
for k = 1:Tmezzi
    P1 = ODFO1.slices(k);
    Vec = size(P1.H);
    for j = 1:Vec(1)
        vt1 = repelem(0,Tmezzi+1);
        for yetanother = 1:k
        vt1(yetanother) = P1.H(j,1);
        end
        vt1(k+1) = P1.H(j,2);
        constraints = [constraints, vt1*(psum1)' <= P1.H(j,3)];
    end
 Q(k+1) = decay*Q(k) + eta *max(0,psum1(k+1));
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
 end 
 for k = 1:Tmezzi
    P1 = ODFO2.slices(k);
    Vec = size(P1.H);
    for j = 1:Vec(1)
        vt1 = repelem(0,Tmezzi+1);
        for yetanother = 1:k
        vt1(yetanother) = P1.H(j,1);
        end
        vt1(k+1) = P1.H(j,2);
        constraints = [constraints, vt1*(psum2)' <= P1.H(j,3)];
    end
 Q(Tmezzi+k+1) = decay*Q(Tmezzi+k) + 1/eta * min(0,psum2(k+1));
%   VCost = VCost + Cost*up((TT-1)*U+k);
 %  psum = psum + up(k) - un(k);
 end
else
    for k = Tmezzi+1:T+1
        Q(k) = 0;
        end
        psum1 = psum11;
        psum2 = repelem(0,Tmezzi+1);
        constraints = [constraints, ccost == tariffbuy1(1:Tmezzi) * net11' + tariffbuy2(1:Tmezzi) * net22' + VCost];        
    
end
 %constraints111 = [constraints111, VQmin <= Q <= VQmax]
% constraints = [constraints, repelem(0,Tmezzi+1) <= psum1 <= repelem(Pmax,Tmezzi+1), repelem(eta*Pmin,Tmezzi+1) <= psum2 <= repelem(0,Tmezzi+1)];
 net11 = net1 + psum1(2:Tmezzi+1);
net22 = net2 + psum2(2:Tmezzi+1);
ccost = tariffbuy1(1:Tmezzi) * net11' + tariffbuy2(1:Tmezzi) * net22' + VCost;
%ccost = ccost111;
Q = Q1;
end
end
Solution = optimize(constraints, ccost);
disp(value(ccost));
RMatrix(1,counter) = ccost;
diffload = max(max(Q) - Qmax,0);
Q1 = Q;
%Q1 = repelem(0,T);
%for checkload=1:T
%    Q1(checkload) = max(0,Q(checkload) - Qmax);
%        ccost = ccost + penalty(checkload) * Q1(checkload);
%end
% Q1 = repelem(0,T+1);
% for checkload=1:T
%     Q1(checkload) = max(0,Q(checkload) - Qmax);
% %    net1(checkload) = net1(checkload) - Q1(checkload)
% %        ccost = ccost + 2*abs(tariffbuy (checkload)) * Q1(checkload);
% end
%check = 0;
%for k = 1:Tmezzi+1
%    if isnan(value(Q(k))) == 1
%        check = 1;
%    end
%end
%if check == 1
%    ccost = ccost111;
%end


nonzero = nnz(max(Q-14-10e-6,0)) + nnz(min(Q+10e-6,0));
% aux = nnz(Q1)
% if aux >= 1
% Q2(1) = Q(1)
%   for kk = 1:T
%    Q2(kk+1) = decay*Q2(kk) + eta *up(kk) + 1/eta * un(kk);
%    VCost = VCost + Cost*(up(kk)+un(kk));
%     end
%     constraints2 = [constraints2, min(0,VPmin) <= un <= min(VPmax,0), max(0,VPmin) <= up <= max(0,VPmax), VQmin <= Q2 <= VQmax];
%     net2 = net + up + un;
%     ccost2 = tariffbuy(1:T) * net2' + VCost;
% Solution2 = optimize(constraints2, ccost2);
% ccost = ccost2
% for checkload=1:T
% ccost = ccost + abs(penalty (checkload)) * Q1(checkload)
% end
% end
% %ccost = tariffbuy(1:T) * net1' + VCost;
% %ccost = ccost + max(diffload-Q(T+1),0) * abs(tariffbuy(T));
% Q(T+1) = max(Q(T+1) - diffload,0);
if value(nonzero) >= 1

Q1 = Q;
nettrue = net1;
Q1(1) = min(Qmax,Q(1));
for checkload = 2:T+1
    checkload2 = checkload-1;
    Q1(checkload) = max(0,min(14,Q(checkload)));
nettrue(checkload) = (Q1(checkload) - Q1(checkload2))*eta^(-sign(Q1(checkload) - Q1(checkload2))) ;
end
net11 = nettrue(2:T+1);
ccost = penalty(1:T) * net11' + VCost;
end
%disp(value(net1));
%constraints = [VPmin <= un <= 0, 0 <= up <= VPmax, VQmin <= Q <= VQmax];%, norm(un.*up) <= 0.01];
%%% IMBALANCE CALCULATION %%%
% upp = sdpvar(1,T);
% unn = sdpvar(1,T);
% Qpn = sdpvar(1,T+1);
% Qpn(1) = Q0;
% for k = 1:Tmezzi
%    Qpn(k+1) = decay*Qpn(k) + eta *upp(k);
%    constraintspn = [unn(k) == 0];
%     end
%     for k = Tmezzi+1:T
%            Qpn(k+1) = decay*Qpn(k) + 1/eta * unn(k);
%     constraintspn = [constraintspn, upp(k) == 0];
%     end
%     constraintspn = [constraintspn, min(0,VPmin) <= unn <= min(VPmax,0), max(0,VPmin) <= upp <= max(0,VPmax), VQmin <= Qpn <= VQmax];
%     net1212 = upp + unn;

% Get the approximate energy schedule 
approxSchedule = value(psum(Tmezzi+2:T+1));

% Compute the imbalances
imbalanceAmount = abs(F1.getEnergyVars() - approxSchedule);
imbalanceCost = sum(penalty(Tmezzi+1:T) .* imbalanceAmount);
optimize(F1.model.constraints, imbalanceCost);
% imbalanceAmount = abs(net1212 - approxSchedule);
% imbalanceCost = sum(max(penalty(1:T),0) .* imbalanceAmount);
% optimize(constraintspn, imbalanceCost);

% Extract results
totalImbalanceAmount = sum(value(imbalanceAmount)); 
totalImbalanceCost = sum(value(imbalanceCost));

% Write down the results
RMatrix(19,counter) = totalImbalanceAmount;
RMatrix(20,counter) = totalImbalanceCost;

%%% IMBALANCE CALCULATION %%%
%ccost = tariffbuy * net1' + VCost;

%optimize(constraints, ccost);
%disp(value(VPmin));
%disp(value(VPmax));
%disp(value(max(0,VPmin) <= up <= max(0,VPmax)));
%disp(value(Type));
disp(value(ccost));
%disp(value(psum));
disp(value(Q));
disp('Counter is');
disp(value(counter));
disp('Parallel is');
disp(value(parallel));
RMatrix(2,counter) = ccost;
RMatrix(3,counter) = nonzero;
for uu = 1:12
    RMatrix(3+uu,counter) = Q(uu);
end
RMatrix(21,counter) = RMatrix(1,counter)+totalImbalanceCost;
QStatus(counter+1) = max(min(Q1(T+1),Qmax-0.001),0);
    end
    filename = sprintf('/home/ubuntu/chargedischarge3_Q0is0_Tis12_Typeis0_051_daysare1to372part%d.xlsx',parallel);
%filename = sprintf('C:\Users\Fab\Documents\Postdoc\chargedischarge3_Q0is0_Tis12_Typeis10_01_daysare1to372part1.xlsx');
writematrix(RMatrix,filename)
%end
end