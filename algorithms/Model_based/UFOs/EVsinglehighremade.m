format shortG
yalmip('solver','glpk');
sett = sdpsettings('verbose',0);

%Program definition
cloud = 1; %set to 1 for using it in cloud
par = 8; %number of iterations going in parallel
dif = 50; %threshold for price differences

if cloud == 0
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\M.mat');
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\N.mat');
elseif cloud == 1
    load('/home/ubuntu/MATLABfiles/M.mat');
    load('/home/ubuntu/MATLABfiles/N.mat');
end

%only take high penalties
aa = (N-M > dif);
bb = find(aa == 1);
M1 = M(bb);
N1 = N(bb);

hour = 17;

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
for pt1 = [100 95 60]
parfor parallel = 1:par %change to "for" or "parfor" 

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
s1 = size(probability1);
s2 = size(probability2);
Ttotal = T0 + s1(2) + s2(2);
t1 = min(find(probability1 > pt))-1;
t2 = min(find(probability2 > 1-pt))-1;

EE = 95;
EUnc = EE/100; %existence probability weekdays
EW = 20;
EWnc = EW/100; %existence probability weekend

days = ceil(38/par);%number of days for iteration
Length = floor(days); %number of cycles needed

RMatrix = repelem(0,19,Length); %initializing result matrix
daysahead = 0; %how many days ahead we want to start; mainly for debugging

if par == 8
    RMatrix(1:7,Length) = repelem(0,7,1);
    Length = Length - 1;
end

for counter = 1:Length
    instance1 = rand;
    instance2 = rand;
    existence = rand;
    if mod(counter,7) > 4
        exist = (existence < EWnc);
    else
        exist = (existence < EUnc);
    end
    i1 = min(find(probability1 > instance1))-1;
    i2 = min(find(probability2 > 1-instance2))-1;
    
    Qmin = Qmin0-5+10*rand;
    QFO = Qmin0-5+10*pt;
    Q0 = Qmin; %initial state of charge for the cycle
    spos = 24*(counter - 1) + 24*Length*(parallel-1)+hour;        
    spotprices = M1(1+spos+daysahead:Ttotal+spos+daysahead,1:1)'; %spot prices
    imbalanceprices = N1(1+spos+daysahead:Ttotal+spos+daysahead,1:1)'; %imbalance prices

    ExactVector = [repelem(0,i1),repelem(1,Ttotal-i1-i2),repelem(0,i2)];
    UFOVector = [repelem(0,t1),repelem(1,Ttotal-t1-t2),repelem(0,t2)];
    DFOVector = [repelem(0,s1(2)-1),repelem(1,Ttotal-s1(2)-s2(2)+2),repelem(0,s2(2)-1)];
    
    data1 = struct();
    data1.InitialSoC = Qmin;
    data1.L = L;
    data1.Pmin = Pmin;
    data1.Pmax = Pmax;
    data1.Qmin = Qmin;
    data1.Qmax = Qmax;
    data1.percentage = 100;
    data1.partial = ExactVector;
    
    F2 = DFOSystem(F1,0.311,data1);
    ActualEnergy = (Qmax - Q0)/L;
    constraints = [F2.getAllConstraints,sum(F2.getEnergyVars()) == ActualEnergy];
    costbattery = spotprices(1:Ttotal) * F2.getEnergyVars()';
    optimize(constraints,costbattery,sett);
    aggrcost = value(costbattery);
    if exist == 0
        aggrcost = 0;
    end
    EnergyValuesUnion = value(F2.getEnergyVars); %Save the energy values
    data2 = data1;

    %Calculate profit - DFO
    
    %create constraints
    data1.partial = DFOVector;
    data1.Qmin = Qmin0-5;
    data1.InitialSoC = Qmin0-5;
    DFO = DFOSystem(F1,0.311,data1);
    DFOcost = spotprices(1:Ttotal) * transpose(DFO.getEnergyVars());
    DFOpenalty = 0;
    DFOEnergy = (Qmax-Qmin0)/L;
    
    %optimize
    DFOSolution = optimize([DFO.getAllConstraints(),sum(DFO.getEnergyVars()) == (Qmax-Qmin0-5)/L],DFOcost,sett);
    if exist == 0
        DFOcost = 0;
        DFOpenalty = imbalanceprices(1:Ttotal) * transpose(DFO.getEnergyVars);
    end
    DFOEnergyValue = value(DFO.getEnergyVars);
    DFOcostValue = value(DFOcost);
    
    %imbalance
if exist == 1
    PenaltyVector = max(ExactVector - DFOEnergyValue/Pmax,0);
    data2.partial = PenaltyVector;
    F2 = DFOSystem(F1,0.311,data2);
    DFOImbCost = imbalanceprices(1:Ttotal) * transpose(F2.getEnergyVars());
    DFOSolutionImb = optimize([F2.getAllConstraints(),sum(F2.getEnergyVars()) == abs(ActualEnergy-DFOEnergy)],DFOImbCost,sett);
    DFOpenalty = DFOpenalty + value(DFOImbCost);
    DFOImbEn = value(F2.getEnergyVars);
end
    DFOTotalCostValue = DFOcostValue + DFOpenalty;
    
    %Calculate profit - UFObase
    
    %create constraints
    km = EUnc;
    km1 = 1;
    pen = 0;
    data1.Pmax = Pmax;
    data1.Qmin = QFO;
    data1.InitialSoC = QFO;
    if mod(counter,7) > 4
        data1.partial = EWnc*UFOVector;
        km = EWnc;
        km1 = EWnc;
        if exist == 1
            pen = 1;
        end
    else
        data1.partial = EUnc*UFOVector;
    end
    UFO = DFOSystem(F1,0.311,data1);
    UFOcost = spotprices(1:Ttotal) * transpose(UFO.getEnergyVars());
    UFOEnergy = (Qmax-QFO)/L;
    
    %optimize
    UFOSolution = optimize([UFO.getAllConstraints(),sum(UFO.getEnergyVars) == km*UFOEnergy],UFOcost,sett);

    UFOpenalty = 0;
    if exist == 0
        UFOcost = 0;
        UFOpenalty = imbalanceprices(1:Ttotal) * transpose(UFO.getEnergyVars);
    end
    if pen == 1
        UFOpenalty = imbalanceprices(1:Ttotal) * transpose(UFO.getEnergyVars) * (1-EWnc)/EWnc;
    end
    UFOcostValue = value(UFOcost);
    UFOEnergyValue = value(UFO.getEnergyVars);

    %imbalance
if (exist == 1 && pen == 0)
    PenaltyVector = max(ExactVector - UFOEnergyValue/Pmax,0);
    data2.partial = PenaltyVector;
    F2 = DFOSystem(F1,0.311,data2);
if ActualEnergy > km*UFOEnergy
imbtariff = imbalanceprices(1:Ttotal);
else
imbtariff = imbalanceprices(1:Ttotal)-spotprices(1:Ttotal);
end
    UFOImbCost = imbtariff * transpose(F2.getEnergyVars());
    UFOSolutionImb = optimize([F2.getAllConstraints(),sum(F2.getEnergyVars()) == abs(ActualEnergy-km*UFOEnergy)],UFOImbCost,sett);
    UFOpenalty = UFOpenalty + value(UFOImbCost);
    UFOImbEn = value(F2.getEnergyVars);
end
    UFOcostValueTot = UFOcostValue + UFOpenalty;
    
    %Calculate profit - UFOreal
    
    %create constraints
    
    data1.partial = UFOVector;
    data1.Qmin = QFO;
    data1.InitialSoC = QFO;
    UFOR = DFOSystem(F1,0.311,data1);
    UFORcost = spotprices(1:Ttotal) * transpose(UFOR.getEnergyVars());
    
    %optimize
    corrfactor = (Qmax-Qmin0-5)/(Qmax-QFO);
    UFOSolution = optimize([UFOR.getAllConstraints(),sum(UFOR.getEnergyVars) == km1*corrfactor*UFOEnergy],UFORcost,sett);

    UFORpenalty = 0;
    if exist == 0
        UFORcost = 0;
        UFORpenalty = imbalanceprices(1:Ttotal) * transpose(UFOR.getEnergyVars);
    end
    if pen == 1
        UFORpenalty = imbalanceprices(1:Ttotal) * transpose(UFOR.getEnergyVars) * (1-EWnc)/EWnc;
    end
    
    UFORcostValue = value(UFORcost);
    UFOREnergyValue = value(UFOR.getEnergyVars);

    %imbalance
if (exist == 1 && pen == 0)
    PenaltyVector = max(ExactVector - UFOREnergyValue/Pmax,0);
    data2.partial = PenaltyVector;
    F2 = DFOSystem(F1,0.311,data2);
if ActualEnergy > km1*UFOEnergy
imbtariff = imbalanceprices(1:Ttotal);
else
imbtariff = imbalanceprices(1:Ttotal)-spotprices(1:Ttotal);
end
    UFORImbCost = imbtariff * transpose(F2.getEnergyVars());
    UFORSolutionImb = optimize([F2.getAllConstraints(),sum(F2.getEnergyVars()) == abs(ActualEnergy-km1*UFOEnergy)],UFORImbCost,sett);
    UFORpenalty = UFORpenalty + value(UFORImbCost);
    UFORImbEn = value(F2.getEnergyVars);
end
    UFORcostValueTot = UFORcostValue + UFORpenalty;
    
    %%%%% ADD RESULTS
    
    RMatrix(1,counter) = aggrcost;
    RMatrix(2,counter) = DFOcostValue;
    RMatrix(3,counter) = DFOpenalty;
    RMatrix(4,counter) = DFOTotalCostValue;
    RMatrix(5,counter) = UFOcostValue; 
    RMatrix(6,counter) = UFOpenalty;
    RMatrix(7,counter) = UFOcostValueTot;
    RMatrix(8,counter) = UFORcostValue; 
    RMatrix(9,counter) = UFORpenalty;
    RMatrix(10,counter) = UFORcostValueTot;
    
    disp('Counter is');
    disp(value(counter));
    disp('Parallel is');
    disp(value(parallel));
end
    %filename = sprintf('/home/ubuntu/imbalance_Q0is0_Tis6_Typeis0_03_daysare1to372part%d.xlsx',parallel);
    if cloud == 1
        filename = sprintf("/home/ubuntu/EVsingleReducedHighPenalties%d_%d_%d_%dT_%d.xlsx",EE,NumBatteries,parallel,pt1,testcase);
    else
        filename = "C:\Users\Fab\Documents\Postdoc\EVhigh"+T+"_"+NumBatteries+".xlsx";
    end
    writematrix(RMatrix,filename)

end
end
end