format shortG
yalmip('solver','glpk');

%Program definition
testing = 0; %set to 1 for enable testing of the functions
cloud = 0; %set to 1 for using it in cloud
visual = 0; %set to 1 to see 3D graphs

if cloud == 0
    load('C:\Users\Fab\Documents\Postdoc\MATLAB files\M.mat');
    load('c:\Users\Fab\Documents\Postdoc\MATLAB files\N.mat');
elseif cloud == 1
    load('/home/ubuntu/MATLABfiles/M.mat');
    load('/home/ubuntu/MATLABfiles/N.mat');
end

if cloud == 1
    PAR = 12;
else
    PAR = 10;
end

for parallel = 1:20 %change to "for" or "parfor" 

T = 3; %generation/planning horizon 
NumBatteries = 50*(parallel+2); %number of batteries
    
%Parameters of the algorithm
Type = 20 ; %if needed, type of FO/DFO used
limit = 15 ; %number of seconds before aborting a process

%Parameters of the battery
InitialSoC = repelem(7,NumBatteries); %initial state of charge
L = repelem(sqrt(0.9),NumBatteries); %loss of the battery (square root of roundtrip efficiency)
Qmin = repelem(0,NumBatteries); %minimum possible state of charge
Qmax = repelem(14,NumBatteries);%maximum possible state of charge
Pmin = repelem(-5,NumBatteries); %maximum discharging power
Pmax = repelem(5,NumBatteries); %maximum charging power
decay = repelem(1,NumBatteries); %decay at each time unit
percentage = 100; %percentage of flexibility exploited by DFO


    
if visual == 1
   T = 3; %set 3D if visualization is wanted 
end

days = 3;%number of days for iteration
Length = 5;%floor(days*24/T); %number of cycles needed
if testing == 1
    Length = 1;
end


RMatrix = repelem(0,19,Length); %initializing result matrix
SoC0 = repelem(0,NumBatteries,Length+1); %initializing "continuity array", 
          %it describes the state of charge at the beginning of next cycle
SoC0(:,1) = InitialSoC'; %initial state of charge
daysahead = 0; %how many days ahead we want to start; mainly for debugging
for counter = 1:Length
      timerzero = tic;
    Q0 = SoC0(:,counter); %initial state of charge for the cycle
    spos = T*(counter - 1)+ T*Length*(parallel-1); %for adjusting time depending on 
                                               %the "counter" and "parallel" values
    constraints = []; %initializing constraints

    Q = sdpvar(NumBatteries,T+1); %state of charge at each time unit between 1 and T+1

    Q(:,1) = Q0; %state of charge at time 1 (redundant?)
    VCost = 0; %optional added cost

    spotprices = M(1+spos+daysahead:T+spos+daysahead,1:1)'; %spot prices
    imbalanceprices = N(1+spos+daysahead:T+spos+daysahead,1:1)'; %imbalance prices

    %PARAMETERS FOR TESTING - to be adapted to this context

    if testing == 1 
        Q0 = [0,0,1];
        L = [1,1,1];
        Qmin = [0 0 0];
        Qmax = [1,3,1];
        Pmin = [-0.01,-0.01,-1];
        Pmax = [1,1,1];
        T = 3;
        decay = [1,1,1];
        spotprices = [-1 1 -1];
    end
 
    %%%%% Instantiate LTI models, one for each battery
    
    k = 1;
    A = [decay(k)];
    B3 = [L(k),1/L(k)]; %FOR DFOS ONLY
    f = [0];
    C = [0];
    D1 = [1,1]; %FOR DFOS ONLY
    PminV = [0;L(k)*Pmin(k)]; %FOR DFOS ONLY
    PmaxV = [Pmax(k);0]; %FOR DFOS ONLY
    lti = LTISystem('A', A, 'B', B3, 'C', C, 'D', D1); %FOR DFOS ONLY
    lti.u.min = PminV;
    lti.u.max = PmaxV;
    lti.x.min = Qmin(k);
    lti.x.max = Qmax(k);
    lti.initialize(Q0(k));

    P = Polyhedron('lb', [Q0(k)], 'ub', [Q0(k)]);
    lti.x.with('initialSet');
    lti.x.initialSet = P;

    lti.instantiate(T);
    F1 = FlexSystem(lti);
    disp("Begin and LTI");
    toc(timerzero)
    timerzero = tic;
    ODFO = DFOSystem(F1,0);
    disp("DFO");
    toc(timerzero)
    timerzero = tic;
    F1Poly = F1.getEnergyPolyhedron;
    LTImodels = repelem(F1,NumBatteries);
    LTIpolys = repelem(F1Poly,NumBatteries);
    DFOmodels = repelem(ODFO,NumBatteries);
    disp("Create the vectors");
    toc(timerzero)
    
    %instantiate a big LTI model, comprehensive of all the batteries
    
    timerzero = tic;
    A = eye(NumBatteries);
    B3 = repelem(0,NumBatteries,2*NumBatteries);
    for k = 1 : NumBatteries
        B3(k,2*k-1) = L(k);
        B3(k,2*k) = 1/L(k);
    end
    f = repelem(0,NumBatteries,NumBatteries);
    C = repelem(0,NumBatteries);
    D1 = repelem(1,2*NumBatteries);
    PminV = [];
    PmaxV = [];
    for k = 1:NumBatteries
        PminV = [PminV;0;L(k)*Pmin(k)]; %FOR DFOS ONLY
        PmaxV = [PmaxV;Pmax(k);0]; %FOR DFOS ONLY
    end
    ltiT = LTISystem('A', A, 'B', B3, 'C', C, 'D', D1); %FOR DFOS ONLY
    ltiT.u.min = PminV;
    ltiT.u.max = PmaxV;
    ltiT.x.min = Qmin';
    ltiT.x.max = Qmax';
    ltiT.initialize(Q0);

    P = Polyhedron('lb', [Q0], 'ub', [Q0]);
    ltiT.x.with('initialSet');
    ltiT.x.initialSet = P;

    ltiT.instantiate(T);
    F1T = FlexSystem(ltiT);
    disp("Create the big model");
    toc(timerzero)
    
    %%%%% AGGREGATION
    
    %Aggregation - Minkowski sum
    
    timerzero = tic;
    AggregateLTI = LTIpolys(1);
    Num = size(LTIpolys);
    time00 = tic;
    for k = 1 : Num(2)-1
        AggregateLTI = AggregateLTI + LTIpolys(k+1); %sum all the polyhedra, one by one
    end
    timeAggregateMinkovski = toc(time00);
    disp("Minkowski aggregation")
    toc(timerzero)
    
    %Aggregation - approximate Minkowski sum
    
    timerzero = tic;
    LTIpolys = num2cell(LTIpolys)'; %convert to cell format
    time0 = tic;
    ApproxMinkowski = MK_outer_approx(LTIpolys); %apparently, there was a built-in function for this
    timeAggregateAM = toc(time0);
    disp("Approximate Minkowski aggregation");
    toc(timerzero)
    
    %Aggregation - DFO
    
    timerzero = tic;
    AggregateDFO = DFOmodels(1).slices;
    ODFO = DFOSystem(DFOmodels(1).slices);
    time0 = tic;
    for k = 1 : NumBatteries - 1 %aggregate one DFO at a time
        [~,~,AggregateDFO] = aggregate_slices_determ3(ODFO,AggregateDFO,DFOmodels(k+1).slices,3);
%        ODFO = DFOSystem(AggregateDFO,F1.getEnergyVars()); %convert it back to a DFO system
    end
    timeAggregateDFO = toc(time0);
    disp("DFO aggregation")
    toc(timerzero)
    
    %%%%% PROFIT CALCULATION
    
    %Calculate profit - non-aggregated LTI models
    timerzero = tic;
    constraints = F1T.getAllConstraints;
    time0 = tic;
    costbattery = spotprices(1:T) * F1T.getEnergyVars()';
    optimize(constraints,costbattery);
    timeOptLTIUnion = toc(time0);
    aggrcost = value(costbattery);
    EnergyValuesUnion = value(F1T.getEnergyVars); %Save the energy values for calculating Q
    DisaggEnergy = disaggregate(F1T,LTImodels);
    disp("Profit NotAggregated")
    toc(timerzero)
    %disaggregation will be needed for later, when computing Q
    
    %Calculate profit - Aggregate LTI
    
    timerzero = tic;
    if ~isempty(AggregateLTI)  %make sure aggregation can be done in time first       
    Representation = AggregateLTI.H; %convert polyhedron to constraints
    ectr = [];
    for k = 1:length(Representation)
        Rep = Representation(k,:);
        ectr = [ectr, Rep(1:T)*F1T.getEnergyVars()' <= Rep(T+1)];
    end
    eccost = spotprices(1:T) * transpose(F1T.getEnergyVars()) + VCost;
    time0 = tic;   
    eSolution = optimize(ectr, eccost);
    timeOptMinkowski = toc(time0);
    MinkowskiCost = value(eccost);
    EnergyValues = value(F1T.getEnergyVars); 
    end
    disp("Profit Minkowski");
    toc(timerzero)
    
    %Calculate profit - Approximate Minkowski
    timerzero = tic;
    Representation = ApproxMinkowski.H; %convert polyhedron to constraints
    ectr = []; 
    for k = 1:length(Representation)
        Rep = Representation(k,:);
        ectr = [ectr, Rep(1:T)*F1T.getEnergyVars()' <= Rep(T+1)];
    end
    eccost = spotprices(1:T) * transpose(F1T.getEnergyVars()) + VCost;
    time0 = tic;
    eSolution = optimize(ectr, eccost);
    timeOptAMinkowski = toc(time0);
    AMinkowskiCost = value(eccost);
    EnergyValuesAM = value(F1T.getEnergyVars); 
    %imbalance calculation
    constraints = F1T.getAllConstraints;
    Imbalance = imbalance_calculation(F1T.getEnergyVars(),EnergyValuesAM,imbalanceprices(1:T),constraints);
    AMEnergyImbalance = Imbalance(1);
    AMCostImbalance = Imbalance(2);
    AMTotalCost = AMinkowskiCost + AMCostImbalance;
    disp("Profit ApproxMinkowski");
    toc(timerzero)
    %Calculate profit - DFO
    timerzero = tic;
    DFOcost = spotprices(1:T) * transpose(ODFO.getEnergyVars()) + VCost;
    time0 = tic;
    DFOSolution = optimize(ODFO.getAllConstraints(),DFOcost);
    timeOptDFO = toc(time0);
    DFOcostValue = value(DFOcost);
    DFOEnergyValue = value(ODFO.getEnergyVars);
    %imbalance calculation
    constraints = F1T.getAllConstraints;
    Imbalance = imbalance_calculation(ODFO.getEnergyVars(),DFOEnergyValue,imbalanceprices(1:T),constraints);
    DFOEnergyImbalance = Imbalance(1);
    DFOCostImbalance = Imbalance(2);
    DFOTotalCost = DFOcostValue + DFOCostImbalance;
    disp("Profit DFO");
    toc(timerzero)
    % DETERMINE SoC
    
    timerzero = tic;
    Q(:,1) = Q0;
    for t = 1:T
        for k = 1:NumBatteries
            Energy = LTImodels(k).getEnergyValues;
            Q(k,t+1) = decay(k)*Q(k,t) + L(k)*max(Energy(t),0) + (1/L(k))*min(Energy(t),0);
        end
    end
    
    for k = 1:NumBatteries
        SoC0(k,counter+1) = max(0,min(Qmax(k),Q(k,T+1)));
    end
    disp("Determine SoC");
    toc(timerzero)

    %%%%% ADD RESULTS
    
    RMatrix(1,counter) = timeAggregateMinkovski;
    RMatrix(2,counter) = timeAggregateAM;
    RMatrix(3,counter) = timeAggregateDFO;
    
    RMatrix(5,counter) = timeOptLTIUnion;
    RMatrix(6,counter) = timeOptMinkowski;
    RMatrix(7,counter) = timeOptAMinkowski;
    RMatrix(8,counter) = timeOptDFO;
    
    RMatrix(10,counter) = aggrcost;
    RMatrix(11,counter) = MinkowskiCost;
    RMatrix(12,counter) = AMinkowskiCost;
    RMatrix(13,counter) = DFOcostValue;
    
    RMatrix(15,counter) = AMCostImbalance;
    RMatrix(16,counter) = DFOCostImbalance;
    
    RMatrix(18,counter) = AMTotalCost;
    RMatrix(19,counter) = DFOTotalCost;
    
    
    disp('Counter is');
    disp(value(counter));
    disp('Parallel is');
    disp(value(parallel));
end
    %filename = sprintf('/home/ubuntu/imbalance_Q0is0_Tis6_Typeis0_03_daysare1to372part%d.xlsx',parallel);
    if cloud == 1
        filename = sprintf("/home/ubuntu/AutoTest%d_%d+.xlsx",T,NumBatteries);
    else
        filename = "C:\Users\Fab\Documents\Postdoc\OnceAgain"+T+"_"+NumBatteries+".xlsx";
    end
    writematrix(RMatrix,filename)

end