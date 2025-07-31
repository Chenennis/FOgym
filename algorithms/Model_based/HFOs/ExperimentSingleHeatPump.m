format shortG
yalmip('solver','glpk');
sett = sdpsettings('verbose',0);

%Program definition
cloud = 0; %set to 1 for using it in cloud
par = 8; %number of iterations going in parallel
time = 900; %length of each time unit, in seconds
daystotal = 365; %number of days that we are considering
T = 12; %generation/planning horizon 

if cloud == 0
    load('MATLAB files\M.mat');
    load('MATLAB files\N.mat');
elseif cloud == 1
    load('/home/ubuntu/MATLABfiles/M.mat');
    load('/home/ubuntu/MATLABfiles/N.mat');
end

M1 = repelem(M,3600/time);
N1 = repelem(N,3600/time);

%instantiate placeholder model

k1 = 1;
A = 1;
B3 = 1; %FOR DFOS ONLY
f = [0];
C = [0];
D1 = [1]; %FOR DFOS ONLY
PminV = [0]; %FOR DFOS ONLY
PmaxV = [1]; %FOR DFOS ONLY
lti = LTISystem('A', A, 'B', B3, 'C', C, 'D', D1); %FOR DFOS ONLY
lti.u.min = PminV;
lti.u.max = PmaxV;
lti.x.min = 0;
lti.x.max = 2;
lti.initialize(0.1);

P = Polyhedron('lb', [0.1], 'ub', [0.1]);
lti.x.with('initialSet');
lti.x.initialSet = P;

lti.instantiate(3);
F1 = FlexSystem(lti);

for testcase = 1:1
for pt1 = 100
for parallel = 1:par %change to "for" or "parfor" 

    
%Parameters of the algorithm
NumSamples = 3 ; %if needed, type of FO/DFO used
NumBatteries = 1;

%Parameters of the HP
Area = 12; %surface area of the room, in square meters
Volume = 60;
c_ht = 0.006; %overall heat transfer coefficient multiplied for time, in kW/(m²K)
c = 1.005; %specific heat capacity of air, in kJ/(kg*K)
m = 1.225*Volume; %mass of the air inside the room, in kg
Pmax = 4.6;
Tlow  = 298;
Thigh = 302;
T_out = 280; % Ambient temp, K
T_in = 300; % Inner starting temperature, K
COP = 3.65; % coefficient of performance
k = 1 - exp(-(Area*c_ht*time)/(c*m));
percentage = 100; %percentage of flexibility exploited by DFO
pt = pt1/100; %probability threshold
gr = 100; %granularity

data = struct();
data.Area = Area;
data.c_ht = c_ht;
data.c = c;
data.m = m;
data.Pmax = Pmax;
data.Tlow = Tlow;
data.Thigh = Thigh;
data.T_in = T_in;
data.T_out = T_out;
data.time = time;
data.T = T;
data.percentage = 100;

days = ceil(daystotal/par);%number of days for iteration
Length = floor(days*24/T*3600/time); %number of cycles needed

RMatrix = repelem(0,19,Length); %initializing result matrix
daysahead = 0; %how many days ahead we want to start; mainly for debugging

for counter = 1:Length
   
    spos = 24*(counter - 1) + 24*Length*(parallel-1);        
    spotprices = M1(1+spos+daysahead:T+spos+daysahead,1:1)'; %spot prices
    imbalanceprices = N1(1+spos+daysahead:T+spos+daysahead,1:1)'; %imbalance prices
    spotpricesCOP = max(spotprices(1:T)/COP,repelem(0.0001,T));
    imbalancepricesCOP = max(imbalanceprices(1:T)/COP,repelem(0.0001,T));
    prices = struct();
    prices.spotprices = spotpricesCOP;
    prices.imbalanceprices = imbalancepricesCOP;
  
    %Model - LTI
    
    A = [1 - k];
    B = [ k/(Area*c_ht)];
    f = [ k*T_out ];
    C = [0];
    D = [1];
    lti = LTISystem('A', A, 'B', B, 'C', C, 'D', D, 'f', f);
    lti.u.min = 0;
    lti.u.max = Pmax;
    lti.x.min = Tlow;
    lti.x.max = Thigh;
    lti.initialize(T_in);
    lti.instantiate(T);
    F = FlexSystem(lti);
    
%     data1 = struct();
%     data1.InitialSoC = T_in;
%     data1.L = 1-k;
%     data1.Pmin = 0;
%     data1.Pmax = Pmax;
%     data1.Qmin = Tlow;
%     data1.Qmax = Thigh;
%     data1.percentage = 100;
%     data1.partial = 1;
    
%    Tk = T_in;
%    Fpartial = F.getEnergyVars();
%    for t = 1:T
%        Tk = A*Tk + B*Fpartial(t) + f;
%    end
%    Tend = Tk;
%    
    
    %Calculate profit - exact
    
%     evars = F.getEnergyVars();
%     Temp = repelem(0,T+1);
%     Temp(1) = T_in;
%     for k = 1:T
%         Temp(k+1) = temp_calc(data,Temp(k),evars(k));
% %        Temp(k+1) = data.Tlow + (data.T_out - data.Tlow + data.Pmax/(data.Area*data.c_ht))*(1-exp(-(data.Area*data.c_ht*(evars(k) - data.Area*data.c_ht*(data.Tlow-data.T_out)*(data.time-(data.c * data.m)/(data.Area*data.c_ht) * log((Temp(k)-data.T_out)/(data.Tlow-data.T_out))))/(data.Pmax - data.Area*data.c_ht*(data.Tlow-data.T_out)))/(data.c*data.m)));
%     end
%     constraints = [repelem(Tlow,T+1) <= Temp <= repelem(Thigh,T+1)]; %Tend == (Tlow+Thigh)/2];
%     costbattery = spotpricesCOP * evars;
%     optimize(constraints,costbattery,sett);
%     EnergyValuesExact = value(evars); %Save the energy values
%     exactcost = value(costbattery);

    %Calculate profit - exact constant
    
    Ttemp = sdpvar(1,T+1);
    Ttemp(1) = T_in;
    ev = sdpvar(1,T);
    for ti = 1:T
        Ttemp(ti+1) = A*Ttemp(ti) + B*ev(ti) + f;
    end
    constraintstruec = [repelem(Tlow,T+1) <= Ttemp <= repelem(Thigh,T+1)];
    cfunc = spotpricesCOP * ev';
    optimize(constraintstruec,cfunc,sett);
    TrueEV = value(ev);
    Truecost = value(cfunc);
    
    
    %Calculate profit - LTI
    
    %optimize
    constraints = F.getAllConstraints; %Tend == (Tlow+Thigh)/2];
    costbattery = spotpricesCOP * F.getEnergyVars()';
    optimize(constraints,costbattery,sett);
    EnergyValuesUnion = value(F.getEnergyVars)*time/3600; %Save the energy values
    aggrcost = value(costbattery);
    
    %imbalance
%     ev = sdpvar(1,T);
%     LTIImbCost = imbalancepricesCOP * transpose(abs(ev-EnergyValuesUnion/time*3600));
%     LTISolutionImb = optimize(constraintstruec,LTIImbCost,sett);
%     LTIpenalty = value(LTIImbCost);
%     LTIImbEn = value(ev);
%     LTITotalCostValue = aggrcost + LTIpenalty;
    
    %Calculate profit - DFO
    
    %create constraints
%    DFO1 = DFOSystem(F,0);
    DFO = DFOSystem(F,0.312,data);
    DFOcost = spotpricesCOP * transpose(DFO.getEnergyVars());
   
    %optimize
    DFOSolution = optimize(DFO.getAllConstraints(),DFOcost,sett);%, TendD == (Tlow+Thigh)/2]
    DFOEnergyValue = value(DFO.getEnergyVars)*time/3600;
    DFOcostValue = value(DFOcost);
    
    %imbalance
    DFOImbCost = imbalancepricesCOP * transpose(abs(F.getEnergyVars()-DFOEnergyValue/time*3600));
    DFOSolutionImb = optimize([F.getAllConstraints()],DFOImbCost,sett);
    DFOpenalty = value(DFOImbCost);
    DFOImbEn = value(F.getEnergyVars);
    DFOTotalCostValue = DFOcostValue + DFOpenalty;
    
    %Calculate profit - exact optimal
    
%     Otemp = sdpvar(1,T+1);
%     Otemp(1) = T_in;
%     evv = sdpvar(1,T);
%     for tj = 1:T
%         Otemp(tj+1) = temp_calc_implicit(data,Otemp(tj),evv(tj));
%     end
%     constraintstrueo = [repelem(Tlow,T+1) <= Otemp <= repelem(Thigh,T+1)];
%     ofunc = spotpricesCOP * evv';
%     optimize(constraintstrueo,ofunc,sett,'solver','bmibnb');
%     TrueOEV = value(evv);
%     TrueOcost = value(ofunc);
    
%    %Calculate profit - HFO1
%    
%     HFO = DFOSystem(F,0.9,data);
%     HFOcost = spotpricesCOP * transpose(HFO.getEnergyVars());
%    
%     %optimize
%     HFOSolution = optimize(HFO.getAllConstraints(),HFOcost,sett);
%     HFOEnergyValue = value(HFO.getEnergyVars);
%     HFOcostValue = value(HFOcost);
%     
%     %imbalance
%     HFOTempCheck = repelem(0,T+1);
%     HFOTempCheck(1) = T_in;
%     for kk = 1:T
%         HFOTempCheck(kk+1) = temp_calc_implicit(data,HFOTempCheck(kk),HFOEnergyValue(kk));
%     end
%     HFOImbCost1 = imbalancepricesCOP * transpose(max(repelem(0,T),HFOTempCheck(2:T+1)-repelem(Thigh,T)))*c*m;
%     HFOImbCost2 = imbalancepricesCOP * transpose(max(repelem(0,T),-HFOTempCheck(2:T+1)+repelem(Tlow,T)))*c*m;
%     HFOImbCost = value(HFOImbCost1 + HFOImbCost2);
%     HFOTotalCostValue = HFOcostValue + HFOImbCost;
%     
%     countchange = repelem(0,T*time/3600);
%     for kk1 = 1:T
%         ind = floor((kk1-1)/4)+1;
%         if HFOTempCheck(kk1) - Tlow > 0.001
%             countchange(ind) = countchange(ind)+1;
%         end
%     end

   %Calculate profit - HFO2
   
    B = (Thigh-Tlow)/(energy_opt(data,Tlow,Thigh)-energy_opt(data,Tlow,Tlow));
    A = B * (energy_opt(data,Tlow,Thigh)-energy_opt(data,Thigh,Thigh))/(Thigh-Tlow);
    f = (1-A)*Tlow - B*energy_opt(data,Tlow,Tlow);
    C = [0];
    D = [1];
    lti2 = LTISystem('A', A, 'B', B, 'C', C, 'D', D, 'f', f);
    lti2.u.min = 0;
    lti2.u.max = Pmax;
    lti2.x.min = Tlow;
    lti2.x.max = Thigh;
    lti2.initialize(T_in);
    lti2.instantiate(T);
    F2 = FlexSystem(lti2);
    
        %Calculate profit - LTI2
    
    %optimize
    constraints = F2.getAllConstraints; %Tend == (Tlow+Thigh)/2];
    costbattery2 = spotpricesCOP * F2.getEnergyVars()';
    optimize(constraints,costbattery2,sett);
    EnergyValuesUnion2 = value(F2.getEnergyVars)*time/3600; %Save the energy values
    aggrcost2 = value(costbattery2);
    
    HFO = DFOSystem(F2,0.9,data);
    HFOcost = spotpricesCOP * transpose(HFO.getEnergyVars());
   
    %optimize
    HFOSolution = optimize(HFO.getAllConstraints(),HFOcost,sett);
    HFOEnergyValue = value(HFO.getEnergyVars);
    HFOcostValue = value(HFOcost);
    
    %imbalance
    %imbalance
    HFOImbCost = imbalancepricesCOP * transpose(abs(F2.getEnergyVars()-HFOEnergyValue));
    HFOSolutionImb = optimize([F2.getAllConstraints()],HFOImbCost,sett);
    HFOpenalty = value(HFOImbCost);
    HFOImbEn = value(abs(F2.getEnergyVars-HFOEnergyValue));
    HFOTotalCostValue = HFOcostValue + HFOpenalty;
    
    HFOTempCheck = repelem(0,T+1);
    HFOTempCheck(1) = T_in;
    for kk = 1:T
        HFOTempCheck(kk+1) = temp_calc_implicit(data,HFOTempCheck(kk),EnergyValuesUnion2(kk)*3600/time);
    end
    
    countchange = repelem(0,T*time/3600);
    for kk1 = 1:T
        ind = floor((kk1-1)/4)+1;
        if abs(HFOTempCheck(kk1+1) - HFOTempCheck(kk1)) > 0.02
            countchange(ind) = countchange(ind)+1;
        end
    end
    
    %candidateperfect1
    
%     evars = HFO.getEnergyVars();
%     Perfectcost = spotpricesCOP * transpose(evars);
%     Tempvar = sdpvar(1,T+1);
%    
%     %optimize
%     constr = [HFO.getAllConstraints()];
%     for tt = 1:T
%         if mod(tt,4) == 2 || mod(tt,4) == 3
%             constr = [constr,evars(tt) == energy_opt(data,Tlow,Tlow)];
%         end
%     end
%     PerfectSolution = optimize(constr,Perfectcost,sett);
%     PerfectEnergyValue = value(evars);
%     PerfectcostValue = value(Perfectcost);
%     
%     %imbalance
%     PTempCheck = repelem(0,T+1);
%     PTempCheck(1) = T_in;
%     for kk = 1:T
%         PTempCheck(kk+1) = temp_calc_implicit(data,PTempCheck(kk),PerfectEnergyValue(kk));
%     end
%     PImbCost1 = imbalancepricesCOP * transpose(max(repelem(0,T),PTempCheck(2:T+1)-repelem(Thigh,T)));
%     PImbCost2 = imbalancepricesCOP * transpose(max(repelem(0,T),-PTempCheck(2:T+1)+repelem(Tlow,T)));
%     PImbCost = value(PImbCost1 + PImbCost2);
%     PTotalCostValue = PerfectcostValue + PImbCost;
    
    %candidate perfect 2

    evars = repelem(0,T);
    Tempvar = repelem(0,T+1);
    evars(1) = energy_opt(data,T_in,Tlow);
    Tempvar (1) = T_in;
    Tempvar (2) = Tlow;
    
    %optimize
    for tt = 2:T
        if mod(tt,4) == 2 || mod(tt,4) == 3
            evars(tt) = energy_opt(data,Tlow,Tlow);
            Tempvar(tt+1) = Tlow;
        elseif mod(tt,4) == 0
            Tempvar(tt+1) = min(max(temp_calc_implicit(data,Tlow,HFOEnergyValue(tt)),Tlow),Thigh); 
            evars(tt) = energy_opt(data,Tlow,Tempvar(tt+1));
        elseif mod(tt,4) == 1
            evars(tt) = energy_opt(data,Tempvar(tt),Tlow);
            Tempvar(tt+1) = Tlow;
        end
    end
    Perfectcost = spotpricesCOP * transpose(evars);
    PerfectEnergyValue = value(evars);
    PerfectcostValue = value(Perfectcost);
    
    %imbalance
    PTempCheck = repelem(0,T+1);
    PTempCheck(1) = T_in;
    for kk = 1:T
        PTempCheck(kk+1) = temp_calc_implicit(data,PTempCheck(kk),PerfectEnergyValue(kk));
    end 
    PImbCost1 = imbalancepricesCOP * transpose(max(repelem(0,T),PTempCheck(2:T+1)-repelem(Thigh,T)))*c*m;
    PImbCost2 = imbalancepricesCOP * transpose(max(repelem(0,T),-PTempCheck(2:T+1)+repelem(Tlow,T)))*c*m;
    PImbCost = value(PImbCost1 + PImbCost2);
    PTotalCostValue = PerfectcostValue + PImbCost;

   %New Q0
   Te = repelem(0,T+1);
   Te(1) = T_in;
   for t = 1:T
       Te(t+1) = A*Te(t) + B*EnergyValuesUnion(t)*3600/time + f;
   end
    T_in = Te(T+1);
    %%%%% ADD RESULTS
    
    RMatrix(1,counter) = aggrcost;
    RMatrix(2,counter) = Truecost;    
    RMatrix(3,counter) = aggrcost2;
    RMatrix(4,counter) = DFOcostValue;
    RMatrix(5,counter) = DFOpenalty;
    RMatrix(6,counter) = DFOTotalCostValue;
    RMatrix(7,counter) = HFOcostValue;
    RMatrix(8,counter) = HFOpenalty;
    RMatrix(9,counter) = HFOTotalCostValue;
    RMatrix(10,counter) = PerfectcostValue;
    RMatrix(11,counter) = PImbCost;
    RMatrix(12,counter) = PTotalCostValue;
    RMatrix(13,counter) = countchange(1);
    RMatrix(14,counter) = countchange(2);
    RMatrix(15,counter) = countchange(3);
    RMatrix(1:12,counter) = RMatrix(1:12,counter)/4;
    
    disp('Counter is');
    disp(value(counter));
    disp('Parallel is');
    disp(value(parallel));
end
    %filename = sprintf('/home/ubuntu/imbalance_Q0is0_Tis6_Typeis0_03_daysare1to372part%d.xlsx',parallel);
    if cloud == 1
        filename = sprintf("/home/ubuntu/HeatHFOsDualLTIcountlong%d_%d.xlsx",parallel,T);
    else
        filename = "C:\Users\Fab\Documents\Postdoc\HeatSimpleDFOcountlong"+T+".xlsx";
    end
    writematrix(RMatrix,filename)

end
end
end
