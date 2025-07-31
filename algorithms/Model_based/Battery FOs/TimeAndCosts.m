format shortG
yalmip('solver','glpk');

%Program definition
testing = 0; %set to 1 for enable testing of the functions
cloud = 0; %set to 1 for using it in cloud
visual = 0; %set to 1 to see 3D graphs

%choice of approaches
UseNoAgg = 0; %no aggregation approach
UseMinkowski = 0; %Minkowski sum
UseAppMink = 0; %approximate Minkowski sum
UseDFO = 1; %DFO

if cloud == 0
    load('C:\Users\Fab\Documents\Postdoc\MATLAB files\M.mat');
    load('c:\Users\Fab\Documents\Postdoc\MATLAB files\N.mat');
elseif cloud == 1
    load('/home/ubuntu/MATLABfiles/M.mat');
    load('/home/ubuntu/MATLABfiles/N.mat');
end


for parallel = 200:100:400 %change to "for" or "parfor" 

T = 5; %generation/planning horizon 
NumBatteries = parallel; %number of batteries; better to keep it even
    
%Parameters of the algorithm
NumSamples = 3 ; %if needed, type of FO/DFO used
limit = 1800 ; %number of seconds before aborting a process
bignumbers = 0; %change to 1 when processing 2000+ batteries

%Parameters of the battery
InitialSoC = 14*rand(1,NumBatteries); %initial state of charge
L = repmat([sqrt(0.9),sqrt(0.965)],1,NumBatteries/2); %loss of the battery (square root of roundtrip efficiency)
Qmin = repelem(0,NumBatteries); %minimum possible state of charge
Qmax = repmat([14,20],1,NumBatteries/2);%maximum possible state of charge
Pmin = repmat([-5,-6.7],1,NumBatteries/2); %maximum discharging power
Pmax = repmat([5,6.7],1,NumBatteries/2); %maximum charging power
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
    Q0 = SoC0(:,1); %initial state of charge for the cycle
    if bignumbers == 1 
        spos = T*(counter - 1);%for adjusting time depending on                        %the "counter" and "parallel" values
    else                       %the "counter" and "parallel" values
        spos = T*(counter - 1) + T*Length*(parallel-1);        
    end
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
    
    %First kind
    k1 = 1;
    A = [decay(k1)];
    B3 = [L(k1),1/L(k1)]; %FOR DFOS ONLY
    f = [0];
    C = [0];
    D1 = [1,1]; %FOR DFOS ONLY
    PminV = [0;L(k1)*Pmin(k1)]; %FOR DFOS ONLY
    PmaxV = [Pmax(k1);0]; %FOR DFOS ONLY
    lti = LTISystem('A', A, 'B', B3, 'C', C, 'D', D1); %FOR DFOS ONLY
    lti.u.min = PminV;
    lti.u.max = PmaxV;
    lti.x.min = Qmin(k1);
    lti.x.max = Qmax(k1);
    lti.initialize(Q0(k1));

    P = Polyhedron('lb', [Q0(k1)], 'ub', [Q0(k1)]);
    lti.x.with('initialSet');
    lti.x.initialSet = P;

    lti.instantiate(T);
    F1 = FlexSystem(lti);
    
    %Second kind
    
    k1 = 2;
    A = [decay(k1)];
    B3 = [L(k1),1/L(k1)]; %FOR DFOS ONLY
    f = [0];
    C = [0];
    D1 = [1,1]; %FOR DFOS ONLY
    PminV = [0;L(k1)*Pmin(k1)]; %FOR DFOS ONLY
    PmaxV = [Pmax(k1);0]; %FOR DFOS ONLY
    lti = LTISystem('A', A, 'B', B3, 'C', C, 'D', D1); %FOR DFOS ONLY
    lti.u.min = PminV;
    lti.u.max = PmaxV;
    lti.x.min = Qmin(k1);
    lti.x.max = Qmax(k1);
    lti.initialize(Q0(k1));

    P = Polyhedron('lb', [Q0(k1)], 'ub', [Q0(k1)]);
    lti.x.with('initialSet');
    lti.x.initialSet = P;

    lti.instantiate(T);
    F2 = FlexSystem(lti);
    
    disp("Begin and LTI");
    toc(timerzero)
    if UseDFO == 1
        timerzero = tic;
        ODFO1 = DFOSystem(F1,0);
        ODFO2 = DFOSystem(F2,0);
        DFOmodels = repmat([ODFO1,ODFO2],1,NumBatteries/2);
        disp("DFO");
        toc(timerzero)
    end
    timerzero = tic;
    LTImodels = repmat([F1,F2],1,NumBatteries/2);    
    if UseMinkowski + UseAppMink > 0
        F1Poly = F1.getEnergyPolyhedron;
        F2Poly = F2.getEnergyPolyhedron;
        LTIpolys = repmat([F1Poly,F2Poly],NumBatteries/2,1);
    else
        timeOptMinkowski = 0;
        timeOptAMinkowski = 0;
        timeMDisaggregate = 0;
        timeAMDisaggregate = 0;
        AMinkowskiCost = 0;
        AMCostImbalance = 0;
        AMTotalCost = 0;
    end
    disp("Create the vectors");
    toc(timerzero)
    
    %instantiate a big LTI model, comprehensive of all the batteries
    if UseNoAgg == 1
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
    timeBigModel = toc(timerzero)
    else
        timeBigModel = 0;
        timeOptLTIUnion = 0;
        timeNoAggDisaggregate = 0;
        aggrcost = 0;
    end
    %%%%% AGGREGATION
    
    %Aggregation - Minkowski sum
    
    if UseMinkowski == 1
        timerzero = tic;
        AggregateLTI = LTIpolys(1);
        Num = size(LTIpolys);
        for k = 1 : Num(1)-1
            AggregateLTI = AggregateLTI + LTIpolys(k+1); %sum all the polyhedra, one by one
        end
        disp("Minkowski aggregation - done")
        toc(timerzero)
    end
    
    %Aggregation - approximate Minkowski sum
    
    if UseAppMink == 1
        timerzero = tic;
        LTIpolys = num2cell(LTIpolys)'; %convert to cell format
        ApproxMinkowski = MK_outer_approx(LTIpolys); %apparently, there was a built-in function for this
        disp("Approximate Minkowski aggregation - done");
        toc(timerzero)
    end
    
    %Aggregation - DFO
    
    if UseDFO == 1
        timerzero = tic;
        AggregateDFO = DFOmodels(1).slices;
        ODFO = DFOmodels(1);
        for k = 1 : NumBatteries - 1 %aggregate one DFO at a time
            [~,~,AggregateDFO] = aggregate_slices_determ3(ODFO,AggregateDFO,DFOmodels(k+1).slices,NumSamples);
        end
        ODFO = DFOSystem(AggregateDFO,F1.getEnergyVars()); %convert it back to a DFO system
        BigDFO = DetDFOSystem(ODFO.slices,DFOmodels);
        disp("DFO aggregation - done")
        timeDFOagg = toc(timerzero)
    end
    
        %Calculate profit - non-aggregated LTI models
    
    if UseNoAgg == 1
        constraints = F1T.getAllConstraints;
        time0 = tic;
        costbattery = spotprices(1:T) * F1T.getEnergyVars()';
        optimize(constraints,costbattery);
        timeOptLTIUnion = toc(time0);
        aggrcost = value(costbattery);
        EnergyValuesUnion = value(F1T.getEnergyVars); %Save the energy values for calculating Q
        
        time0 = tic;
        DisaggEnergy = disaggregate(F1T,LTImodels);
        timeNoAggDisaggregate = toc(time0);
        disp("Profit NotAggregated")
        %disaggregation will be needed for later, when computing Q
    end
    
    %Calculate profit - Minkowski
    %Add one test: do F1T.getEnergyVars behave properly from now on?
    
    if UseMinkowski == 1
        %convert polyhedron to constraints
        Representation = AggregateLTI.H; 
        ectr = []; 
        for k = 1:length(Representation)
            Rep = Representation(k,:);
            ectr = [ectr, Rep(1:T)*F1T.getEnergyVars()' <= Rep(T+1)];
        end
        
        %optimize and disaggregation
        eccost = spotprices(1:T) * transpose(F1T.getEnergyVars());
        time0 = tic;
        eSolution = optimize(ectr, eccost);
        timeOptMinkowski = toc(time0);
        MinkowskiCost = value(eccost);
        EnergyValuesM = value(F1T.getEnergyVars);
        
        time0 = tic;
        DisaggEnergyM = disaggregate(F1T,LTImodels);
        timeMDisaggregate = toc(time0);
        disp("Profit Minkowski");
    end
    
    %Calculate profit - Approximate Minkowski
    
    if UseAppMink == 1
        %convert polyhedron to constraints
        Representation = ApproxMinkowski.H; 
        ectr = []; 
        for k = 1:length(Representation)
            Rep = Representation(k,:);
            ectr = [ectr, Rep(1:T)*F1T.getEnergyVars()' <= Rep(T+1)];
        end
        
        %optimize
        eccost = spotprices(1:T) * transpose(F1T.getEnergyVars());
        time0 = tic;
        eSolution = optimize(ectr, eccost);
        timeOptAMinkowski = toc(time0);
        AMinkowskiCost = value(eccost);
        EnergyValuesAM = value(F1T.getEnergyVars);
        
        %imbalance calculation and disaggregation
        constraints = F1T.getAllConstraints;
        Imbalance = imbalance_calculation(F1T.getEnergyVars(),EnergyValuesAM,imbalanceprices(1:T),constraints);
        AMEnergyImbalance = Imbalance(1);
        AMCostImbalance = Imbalance(2);
        AMTotalCost = AMinkowskiCost + AMCostImbalance;
        
        time0 = tic;
        DisaggEnergyAM = disaggregate(F1T,LTImodels);
        timeAMDisaggregate = toc(time0);
        disp("Profit ApproxMinkowski");
    end
    
    %Calculate profit - DFO
    
    if UseDFO == 1
        %optimize
        DFOcost = spotprices(1:T) * transpose(BigDFO.getEnergyVars());
        time0 = tic;
        DFOSolution = optimize(BigDFO.getAllConstraints(),DFOcost);
        timeOptDFO = toc(time0);
        DFOcostValue = value(DFOcost);
        DFOEnergyValue = value(BigDFO.getEnergyVars);
        
        %imbalance calculation and disaggregation
        constraints = F1T.getAllConstraints;
        Imbalance = imbalance_calculation(F1T.getEnergyVars(),DFOEnergyValue,imbalanceprices(1:T),constraints);
        DFOEnergyImbalance = Imbalance(1);
        DFOCostImbalance = Imbalance(2);
        DFOTotalCost = DFOcostValue + DFOCostImbalance;
        
        time0 = tic;
        [eS, tS, d_err] = disagg_dfo_slices(BigDFO, DFOmodels);
        timeDFODisaggregate = toc(time0);
        disp("Profit DFO");
    end
    
    % DETERMINE SoC
    
    timerzero = tic;
    Q = repelem(0,NumBatteries,T+1);
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
    
    RMatrix(1,counter) = timeBigModel;
    RMatrix(2,counter) = timeOptLTIUnion;
    RMatrix(3,counter) = timeOptMinkowski;
    RMatrix(4,counter) = timeOptAMinkowski;
    RMatrix(5,counter) = timeOptDFO;
    RMatrix(6,counter) = timeDFOagg;
    
    RMatrix(7,counter) = timeNoAggDisaggregate;
    RMatrix(8,counter) = timeMDisaggregate;
    RMatrix(9,counter) = timeAMDisaggregate;
    RMatrix(10,counter) = timeDFODisaggregate;

    RMatrix(12,counter) = aggrcost;
    RMatrix(13,counter) = AMinkowskiCost;
    RMatrix(14,counter) = AMCostImbalance;
    RMatrix(15,counter) = AMTotalCost;
    RMatrix(16,counter) = DFOcostValue;
    RMatrix(17,counter) = DFOCostImbalance;
    RMatrix(18,counter) = DFOTotalCost;
    
    
    disp('Counter is');
    disp(value(counter));
    disp('Parallel is');
    disp(value(parallel));
end
    %filename = sprintf('/home/ubuntu/imbalance_Q0is0_Tis6_Typeis0_03_daysare1to372part%d.xlsx',parallel);
    if cloud == 1
        filename = sprintf("/home/ubuntu/TimeAndCosts%d_%d.xlsx",T,NumBatteries);
    else
        filename = "C:\Users\Fab\Documents\Postdoc\0TimeAndCosts"+T+"_"+NumBatteries+".xlsx";
    end
    writematrix(RMatrix,filename)

end