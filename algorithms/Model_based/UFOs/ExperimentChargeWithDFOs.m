format shortG
yalmip('solver','sedumi');
sett = sdpsettings('verbose',0);

%Program definition
cloud = 0; %set to 1 for using it in cloud
par = 8; %number of iterations going in parallel
prob = [1:-0.1:0.6]; %probabilities I want to consider

if cloud == 0
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\M.mat');
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\N.mat');
elseif cloud == 1
    load('/home/ubuntu/MATLABfiles/M.mat');
    load('/home/ubuntu/MATLABfiles/N.mat');
end

for pt = prob
for parallel = 1:par %change to "for" or "parfor" 

    ptt = floor(pt*100);
T = 12; %generation/planning horizon 
TT = floor(T/2);
    
%Parameters of the algorithm
NumSamples = 3 ; %if needed, type of FO/DFO used
NumBatteries = 1;

%Parameters of the battery
InitialSoC = 0; %initial state of charge
L = sqrt(0.9); %loss of the battery (square root of roundtrip efficiency)
Qmin = 0; %minimum possible state of charge
Qmax = 14;%maximum possible state of charge
Pmin = -5; %maximum discharging power
Pmax = 5; %maximum charging power
decay = 1; %decay at each time unit
percentage = 100; %percentage of flexibility exploited by DFO
%pt = 0.95; %probability threshold
gr = 100; %granularity

days = ceil(365/par);%number of days for iteration
Length = floor(days*24/T); %number of cycles needed

RMatrix = repelem(0,19,Length); %initializing result matrix
SoC0 = repelem(0,NumBatteries,Length+1); %initializing "continuity array", 
          %it describes the state of charge at the beginning of next cycle
SoC0(:,1) = InitialSoC'; %initial state of charge
daysahead = 0; %how many days ahead we want to start; mainly for debugging
Unit = eye(T);

for counter = 1:Length
    Q0 = SoC0(:,counter); %initial state of charge for the cycle
    spos = T*(counter - 1) + T*Length*(parallel-1); 
    spotprices = M(1+spos+daysahead:T+spos+daysahead,1:1)'; %spot prices
    imbalanceprices = N(1+spos+daysahead:T+spos+daysahead,1:1)'; %imbalance prices

    prices = struct();
    prices.spotprices = spotprices;
    prices.imbalanceprices = imbalanceprices;
    %%%%% Instantiate LTI models, one for each battery
    
    %Total
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
    lti.initialize(Q0);
    data1 = struct();
    data1.InitialSoC = Q0;
    data1.L = L;
    data1.Pmin = Pmin;
    data1.Pmax = Pmax;
    data1.Qmin = Qmin;
    data1.Qmax = Qmax;
    data1.percentage = 100;

    P = Polyhedron('lb', [Q0], 'ub', [Q0]);
    lti.x.with('initialSet');
    lti.x.initialSet = P;

    lti.instantiate(T);
    F1 = FlexSystem(lti);
    
    %First
    A = decay;
    B3 = [L]; %FOR DFOS ONLY
    f = [0];
    C = [0];
    D1 = [1]; %FOR DFOS ONLY
    lti = LTISystem('A', A, 'B', B3, 'C', C, 'D', D1); %FOR DFOS ONLY
    lti.u.min = 0;
    lti.u.max = Pmax;
    lti.x.min = Qmin;
    lti.x.max = Qmax;
    lti.initialize(Q0);
    data2 = struct();
    data2.InitialSoC = Q0;
    data2.L = L;
    data2.Pmin = 0;
    data2.Pmax = Pmax;
    data2.Qmin = Qmin;
    data2.Qmax = Qmax;
    data2.percentage = 100;

    P = Polyhedron('lb', [Q0], 'ub', [Q0]);
    lti.x.with('initialSet');
    lti.x.initialSet = P;

    lti.instantiate(T/2);
    F2 = FlexSystem(lti);
        
    %Calculate profit - LTI model
    
    constraints = F1.getAllConstraints;
    evars = F1.getEnergyVars();
    constraints = [constraints, repelem(0,T/2) <= evars(1:T/2), evars(T/2+1:T) <= repelem(0,T/2), Q0 + L*sum(evars(1:T/2)) + 1/L * sum(evars(T/2+1:T)) >= Qmin];
    costbattery = spotprices(1:T) * evars';
    optimize(constraints,costbattery,sett);
    aggrcost = value(costbattery);
    EnergyValuesUnion = value(F1.getEnergyVars); %Save the energy values for calculating Q

            %Second
    Q1 = Q0 + L*(sum(EnergyValuesUnion(1:T/2)));
    A = decay;
    B3 = [1/L]; %FOR DFOS ONLY
    f = [0];
    C = [0];
    D1 = [1]; %FOR DFOS ONLY
    lti = LTISystem('A', A, 'B', B3, 'C', C, 'D', D1); %FOR DFOS ONLY
    lti.u.min = L*Pmin;
    lti.u.max = 0;
    lti.x.min = Qmin;
    lti.x.max = Qmax;
    lti.initialize(Q1);
    data3 = struct();
    data3.InitialSoC = Q1;
    data3.L = L;
    data3.Pmin = Pmin;
    data3.Pmax = 0;
    data3.Qmin = Qmin;
    data3.Qmax = Qmax;
    data3.percentage = 100;

    P = Polyhedron('lb', [Q1], 'ub', [Q1]);
    lti.x.with('initialSet');
    lti.x.initialSet = P;

    lti.instantiate(T/2);
    F3 = FlexSystem(lti);

    %Calculate profit - DFOs
    
    %create constraints
    ODFO1 = DFOSystem(F2,0.3,data2);
    ODFO2 = DFOSystem(F3,0.3,data3);
    c1 = [ODFO1.getAllConstraints(),sum(ODFO1.getEnergyVars) == (Q1 - Q0)/L];
    DFOcost1 = spotprices(1:T/2) * transpose(ODFO1.getEnergyVars());
    DFOcost2 = spotprices(T/2+1:T) * transpose(ODFO2.getEnergyVars());
    
    %optimize
    DFOSolution1 = optimize(c1,DFOcost1,sett);
    DFOSolution2 = optimize(ODFO2.getAllConstraints(),DFOcost2,sett);
    DFOcostValue = value(DFOcost1+DFOcost2);
    DFOEnergyValue = value([ODFO1.getEnergyVars,ODFO2.getEnergyVars]);

    %imbalance calculation and disaggregation
    constraints = F1.getAllConstraints;
    DFOImbalance = imbalance_calculation(F1.getEnergyVars(),DFOEnergyValue,imbalanceprices(1:T),constraints);
    DFOEnergyImbalance = DFOImbalance(1);
    DFOCostImbalance = DFOImbalance(2);
    DFOTotalCost = DFOcostValue + DFOCostImbalance;
    
   %Calculate profit - UFOs
    
    %Calculate profit - ProbDFO
    
    %optimization
    prices1 = struct();
    prices2 = struct();
    prices1.spotprices = spotprices(1:T/2);
    prices2.spotprices = spotprices(T/2+1:T);
    prices1.imbalanceprices = imbalanceprices(1:T/2);
    prices2.imbalanceprices = imbalanceprices(T/2+1:T);
    UFO1 = UFO_2steps(DFOEnergyValue(1:T/2),data2,pt,prices1,sett);
    UFO2 = UFO_2steps(DFOEnergyValue(T/2+1:T),data3,pt,prices2,sett);
    UFOEnergy = [UFO1.energy,UFO2.energy];
    UFOcost = UFO1.cost+UFO2.cost;
    
    %imbalance calculation and disaggregation
        
    ImbalanceP1 = imbalance_calculation_n(F1.getEnergyVars(),UFOEnergy,imbalanceprices(1:T),constraints);
    UFOEnergyImbalance = ImbalanceP1(1);
    UFOCostImbalance = ImbalanceP1(2);
    UFOTotalCost = UFOcost + UFOCostImbalance;

    % DETERMINE SoC
    
    Q = repelem(0,T+1);
    Q(:,1) = Q0;
    for t = 1:T
        Energy = EnergyValuesUnion;
        Q(t+1) = decay*Q(t) + L*max(Energy(t),0) + (1/L)*min(Energy(t),0);
    end
    
    SoC0(1,counter+1) = max(0,min(Qmax,Q(T+1)));
    
    %%%%% ADD RESULTS
    
    RMatrix(1,counter) = aggrcost;
    RMatrix(2,counter) = DFOcostValue;
    RMatrix(3,counter) = DFOCostImbalance;
    RMatrix(4,counter) = DFOTotalCost;     
    RMatrix(5,counter) = UFOcost;
    RMatrix(6,counter) = UFOCostImbalance;
    RMatrix(7,counter) = UFOTotalCost;
    
    
    disp('Counter is');
    disp(value(counter));
    disp('Parallel is');
    disp(value(parallel));
end
    %filename = sprintf('/home/ubuntu/imbalance_Q0is0_Tis6_Typeis0_03_daysare1to372part%d.xlsx',parallel);
    if cloud == 1
        filename = sprintf("/home/ubuntu/22SFOChargingZero%d_%d_%d_pt%d.xlsx",T,NumBatteries,parallel,ptt);
    else
        filename = "C:\Users\Fab\Documents\Postdoc\Disaggregate"+T+"_"+NumBatteries+".xlsx";
    end
    writematrix(RMatrix,filename)

end
end