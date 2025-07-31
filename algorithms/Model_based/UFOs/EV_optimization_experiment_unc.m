format shortG
yalmip('solver','glpk');
sett = sdpsettings('verbose',0);

%Program definition
cloud = 0; %set to 1 for using it in cloud
par = 8; %number of iterations going in parallel

if cloud == 0
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\M.mat');
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\N.mat');
elseif cloud == 1
    load('/home/ubuntu/MATLABfiles/M.mat');
    load('/home/ubuntu/MATLABfiles/N.mat');
end

for testcase = 1:5
for pt1 = [100 95 60]
for parallel = 1:par %change to "for" or "parfor" 

T0 = 10; %generation/planning horizon 
    
%Parameters of the algorithm
NumSamples = 3 ; %if needed, type of FO/DFO used
NumBatteries = 1;

%Parameters of the battery
InitialSoC = 0; %initial state of charge
L = 0.84; %loss of the battery (square root of roundtrip efficiency)
Qmin0 = 30; %minimum possible state of charge
Qmax = 60;%maximum possible state of charge
Pmin = 0; %maximum discharging power
Pmax = 7; %maximum charging power
decay = 1; %decay at each time unit
percentage = 100; %percentage of flexibility exploited by DFO
pt = pt1/100; %probability threshold
gr = 100; %granularity

probability1 = [0.5 0.876 0.966 1.01];
probability2 = [0.08 0.5 1.01];
ss = size(probability2);
t1 = min(find(probability1 > pt))-1;
t2 = min(find(probability2 > 1-pt))-1;
hour = 17+t1; %at which hour does the charging start
EE = 20;
EUnc = EE/100; %existence probability

T = T0 + ss(2) - t1 + t2;

days = ceil(365/par);%number of days for iteration
Length = floor(days); %number of cycles needed

RMatrix = repelem(0,19,Length); %initializing result matrix
daysahead = 0; %how many days ahead we want to start; mainly for debugging
Unit = eye(T);
recover = 0; %amount of charge to be recovered

for counter = 1:Length
    instance1 = rand;
    instance2 = rand;
    existence = rand;
    if mod(counter,7) > 4
        exist = (existence < EUnc);
    else
        exist = 1;
    end
    i1 = min(find(probability1 > instance1))-1;
    i2 = min(find(probability2 > 1-instance2))-1;
    hour2 = 17+i1;
    T2 = T0 + ss(2) + i2 - i1;
    
    Qmin = Qmin0-5+10*rand;
    QFO = Qmin0-5+10*pt;
    Q0 = Qmin-recover; %initial state of charge for the cycle
    spos = 24*(counter - 1) + 24*Length*(parallel-1)+hour;        
    spos2 = 24*(counter - 1) + 24*Length*(parallel-1)+hour2; 
    sposDFO = 24*(counter - 1) + 24*Length*(parallel-1)+20;
    spotprices = M(1+spos+daysahead:T+spos+daysahead,1:1)'; %spot prices
    spotprices2 = M(1+spos2+daysahead:T2+spos2+daysahead,1:1)'; %spot prices for baseline
    imbalanceprices = M(1+spos+daysahead:T+spos+daysahead,1:1)'; %imbalance prices
    spotDFO = M(1+sposDFO+daysahead:T0+sposDFO+daysahead,1:1)'; %spot prices
    imbalanceDFO = M(1+sposDFO+daysahead:T0+sposDFO+daysahead,1:1)'; %imbalance prices

    %%%%% Instantiate LTI models, one for each battery
    
    %First kind
    k1 = 1;
    A = decay;
    B3 = [L,1/L]; %FOR DFOS ONLY
    f = [0];
    C = [0];
    D1 = [1,1]; %FOR DFOS ONLY
    PminV = [0;L*Pmin]; %FOR DFOS ONLY
    PmaxV = [Pmax;0]; %FOR DFOS ONLY
    lti = LTISystem('A', A, 'B', B3, 'C', C, 'D', D1); %FOR DFOS ONLY
    lti.u.min = PminV;
    lti.u.max = PmaxV;
    lti.x.min = Qmin;
    lti.x.max = Qmax;
    lti.initialize(QFO);
    data1 = struct();
    data1.InitialSoC = QFO;
    data1.L = L;
    data1.Pmin = Pmin;
    data1.Pmax = Pmax;
    data1.Qmin = Qmin;
    data1.Qmax = Qmax;
    data1.percentage = 100;

    P = Polyhedron('lb', [Q0], 'ub', [Q0]);
    lti.x.with('initialSet');
    lti.x.initialSet = P;
    
    lti2 = lti.copy;
    ltiD = lti.copy;

    lti.instantiate(T);
    F1 = FlexSystem(lti);
    lti2.instantiate(T2);
    F2 = FlexSystem(lti2);
    ltiD.instantiate(T0);
    FD = FlexSystem(ltiD);
    
    %Calculate profit - exact
    
    constraints = [F2.getAllConstraints,sum(F2.getEnergyVars) == (Qmax - Q0)/L];
    costbattery = spotprices2(1:T2) * F2.getEnergyVars()';
    optimize(constraints,costbattery,sett);
    aggrcost = value(costbattery);
    if exist == 0
        aggrcost = 0;
    end
    EnergyValuesUnion = value(F2.getEnergyVars); %Save the energy values for calculating Q
    
    %Calculate profit - DFO
    
    %create constraints
    DFO = DFOSystem(FD,0.3,data1);
    DFOcost = spotDFO(1:T0) * transpose(DFO.getEnergyVars());
    DFOpenalty = 0;
    
    %optimize
    DFOSolution = optimize([DFO.getAllConstraints(),sum(DFO.getEnergyVars) == (Qmax-Q0)/L],DFOcost,sett);
    if exist == 0
        DFOcost = 0;
        DFOpenalty = imbalanceDFO(1:T0) * transpose(DFO.getEnergyVars);
    end
    DFOcostValue = value(DFOcost);
    DFOTotalCostValue = DFOcostValue + DFOpenalty;
    DFOEnergyValue = value(DFO.getEnergyVars);

    %Calculate profit - UFObase
    
    %create constraints
    km = 1;
    pen = 0;
    if mod(counter,7) > 4
        data1.Pmax = EUnc*Pmax;
        km = EUnc;
        if exist == 1
            pen = 1;
        end
    end
    UFO = DFOSystem(F1,0.3,data1);
    UFOcost = spotprices(1:T) * transpose(UFO.getEnergyVars());
    
    %optimize
    UFOSolution = optimize([UFO.getAllConstraints(),sum(UFO.getEnergyVars) == km*(Qmax-Q0)/L],UFOcost,sett);

    penalty = 0;
    if exist == 0
        UFOcost = 0;
        penalty = imbalanceprices(1:T) * transpose(UFO.getEnergyVars);
    end
    if pen == 1
        penalty = imbalanceprices(1:T) * transpose(UFO.getEnergyVars) * (1-EUnc)/EUnc;
    end
    UFOcostValue = value(UFOcost);
    UFOEnergyValue = value(UFO.getEnergyVars);

     recover = 0;
    if i1 > t1
        for k = 1:i1-t1
            penalty = penalty + UFOEnergyValue(k)*(imbalanceprices(k)-spotprices(k));
        end
    end
    if t2 > i2
        for k = 1:i2-t2
            penalty = penalty + UFOEnergyValue(k)*(imbalanceprices(k)-spotprices(k));
        end
    end
    UFOcostValueTot = UFOcostValue + penalty;
    penalty1 = penalty;
    
    %Calculate profit - UFOreal
    
    %create constraints
    if mod(counter,7) > 4
        data1.Pmax = Pmax;
    end
    
    UFOR = DFOSystem(F1,0.3,data1);
    UFORcost = spotprices(1:T) * transpose(UFOR.getEnergyVars());
    
    %optimize
    UFOSolution = optimize([UFOR.getAllConstraints(),sum(UFOR.getEnergyVars) == km*(Qmax-Q0)/L],UFORcost,sett);

    penalty = 0;
    if exist == 0
        UFORcost = 0;
        penalty = imbalanceprices(1:T) * transpose(UFOR.getEnergyVars);
    end
    if pen == 1
        penalty = imbalanceprices(1:T) * transpose(UFOR.getEnergyVars) * (1-EUnc)/EUnc;
    end
    UFORcostValue = value(UFORcost);
    UFOREnergyValue = value(UFOR.getEnergyVars);

    if i1 > t1
        for k = 1:i1-t1
            penalty = penalty + UFOREnergyValue(k)*(imbalanceprices(k)-spotprices(k));
        end
    end
    if t2 > i2
        for k = 1:i2-t2
            penalty = penalty + UFOREnergyValue(k)*(imbalanceprices(k)-spotprices(k));
        end
    end
    UFORcostValueTot = UFORcostValue + penalty;
    
    %%%%% ADD RESULTS
    
    RMatrix(1,counter) = aggrcost;
    RMatrix(2,counter) = DFOcostValue;
    RMatrix(3,counter) = DFOpenalty;
    RMatrix(4,counter) = DFOTotalCostValue;
    RMatrix(5,counter) = UFOcostValue; 
    RMatrix(6,counter) = penalty1;
    RMatrix(7,counter) = UFOcostValueTot;
    RMatrix(5,counter) = UFORcostValue; 
    RMatrix(6,counter) = penalty;
    RMatrix(7,counter) = UFORcostValueTot;
    
    disp('Counter is');
    disp(value(counter));
    disp('Parallel is');
    disp(value(parallel));
end
    %filename = sprintf('/home/ubuntu/imbalance_Q0is0_Tis6_Typeis0_03_daysare1to372part%d.xlsx',parallel);
    if cloud == 1
        filename = sprintf("/home/ubuntu/EVsinglebase%d_%d_%d_%d_T%d.xlsx",EE,NumBatteries,parallel,pt1,testcase);
    else
        filename = "C:\Users\Fab\Documents\Postdoc\EVsingle"+T+"_"+NumBatteries+".xlsx";
    end
    writematrix(RMatrix,filename)

end
end
end