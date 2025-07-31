format shortG
yalmip('solver','glpk');
sett = sdpsettings('verbose',0);

%Program definition
cloud = 0; %set to 1 for using it in cloud
par = 1; %number of parallel processes
dif = 50; %threshold for price difference

%choice of approaches
UseNoAgg = 0; %no aggregation approach
UseMinkowski = 0; %Minkowski sum
UseAppMink = 0; %approximate Minkowski sum
UseDFO = 0; %DFO
UseUFO = 0; %UFO
UseUncBase = 1;% Uncertain base

if cloud == 0
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\M.mat');
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\N.mat');
elseif cloud == 1
    load('/home/ubuntu/MATLABfiles/M.mat');
    load('/home/ubuntu/MATLABfiles/N.mat');
end
RT = [];
for iter = 1:1
for T = 5%[3 4 5 6 12 18 24 36 48 72 96]%5
for parallel = [2 10 50 100 200 400 700 1000] %change to "for" or "parfor" 

%T = 96; %generation/planning horizon 
NumBatteries = parallel; %number of batteries; better to keep it even
    
%Parameters of the algorithm
NumSamples = 3 ; %if needed, type of FO/DFO used
limit = 1800 ; %number of seconds before aborting a process
bignumbers = 1; %change to 1 when processing 2000+ batteries

%Parameters of the battery
InitialSoC = 0; %initial state of charge
L = repmat([sqrt(0.9),sqrt(0.965)],1,NumBatteries/2); %loss of the battery (square root of roundtrip efficiency)
Qmin = repelem(0,NumBatteries); %minimum possible state of charge
Qmax = repmat([14,20],1,NumBatteries/2);%maximum possible state of charge
Pmin = repmat([-5,-6.7],1,NumBatteries/2); %maximum discharging power
Pmax = repmat([5,6.7],1,NumBatteries/2); %maximum charging power
decay = repelem(1,NumBatteries); %decay at each time unit
percentage = 100; %percentage of flexibility exploited by DFO
%pt = 0.9; %probability threshold

days = floor(119/par);%number of days for iteration
Length = floor(days*24/T); %number of cycles needed

RMatrix = repelem(0,36,Length); %initializing result matrix
SoC0 = repelem(0,NumBatteries,Length+1); %initializing "continuity array", 
          %it describes the state of charge at the beginning of next cycle
SoC0(:,1) = InitialSoC'; %initial state of charge
daysahead = 0; %how many days ahead we want to start; mainly for debugging
for counter = 1:11
    Q0 = SoC0(:,counter); %initial state of charge for the cycle
    if bignumbers == 1 
        spos = T*(counter - 1);%for adjusting time depending on                        %the "counter" and "parallel" values
    else                       %the "counter" and "parallel" values
        spos = T*(counter - 1) + T*Length*(parallel-1);        
    end
    aa = (N-M > dif);
    bb = find(aa == 1);
    M1 = M(bb);
    N1 = N(bb);
    spotprices = M1(1+spos+daysahead:T+spos+daysahead,1:1)'; %spot prices
    imbalanceprices = N1(1+spos+daysahead:T+spos+daysahead,1:1)'; %imbalance prices

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
    
    data1 = struct();
    data1.InitialSoC = Q0(k1);
    data1.L = L(k1);
    data1.Pmin = Pmin(k1);
    data1.Pmax = Pmax(k1);
    data1.Qmin = Qmin(k1);
    data1.Qmax = Qmax(k1);
    data1.percentage = percentage;
    data1.pt = 0.9;

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
    
    data2 = struct();
    data2.InitialSoC = Q0(k1);
    data2.L = L(k1);
    data2.Pmin = Pmin(k1);
    data2.Pmax = Pmax(k1);
    data2.Qmin = Qmin(k1);
    data2.Qmax = Qmax(k1);
    data2.percentage = percentage;
    data2.pt = 0.9;

    P = Polyhedron('lb', [Q0(k1)], 'ub', [Q0(k1)]);
    lti.x.with('initialSet');
    lti.x.initialSet = P;

    lti.instantiate(T);
    F2 = FlexSystem(lti);
    
    data = repmat([data1,data2],1,NumBatteries/2);
    if UseDFO == 1
        ODFO1 = DFOSystem(F1,0.3,data1);
        ODFO2 = DFOSystem(F2,0.3,data2);
        DFOmodels = repmat([ODFO1,ODFO2],1,NumBatteries/2);
    end
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
    timeOptMinkowski = 0;
    timeMDisaggregate = 0;            
    
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
        timeBigModel = toc(timerzero);
    else
        timeBigModel = 0;
        timeOptLTIUnion = 0;
        timeNoAggDisaggregate = 0;
        aggrcost = 0;
        DFOCostImbalance = 0;
        DFOTotalCost = 0;
    end
    %%%%% AGGREGATION
    
    %Aggregation - Minkowski sum
    
    if UseMinkowski == 1
        time0 = tic;
        AggregateLTI = LTIpolys(1);
        Num = size(LTIpolys);
        for k = 1 : Num(1)-1
            AggregateLTI = AggregateLTI + LTIpolys(k+1); %sum all the polyhedra, one by one
        end
        timeMinkowskiAgg = toc(time0);
    end
    
    %Aggregation - approximate Minkowski sum
    
    if UseAppMink == 1
        time0 = tic;
        LTIpolys = num2cell(LTIpolys)'; %convert to cell format
        ApproxMinkowski = MK_outer_approx(LTIpolys); %apparently, there was a built-in function for this
        timeAppMinkAgg = toc(time0);
    end
    
    %Aggregation - DFO
    
    if UseDFO == 1
        timerzero = tic;
        AggregateDFO = DFOSystem(DFOmodels);
        DFOAggTime = toc(timerzero);
    end
    
        %Calculate profit - non-aggregated LTI models
    
    if UseNoAgg == 1
        constraints = F1T.getAllConstraints;
        time0 = tic;
        costbattery = spotprices(1:T) * F1T.getEnergyVars()';
        optimize(constraints,costbattery,sett);
        timeOptLTIUnion = toc(time0);
        aggrcost = value(costbattery);
        EnergyValuesUnion = value(F1T.getEnergyVars); %Save the energy values for calculating Q
        
        time0 = tic;
        DisaggEnergy = disaggregate(F1T,LTImodels);
        timeNoAggDisaggregate = toc(time0);
        
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
            ectr = [ectr, Rep(1:T)*F1.getEnergyVars()' <= Rep(T+1)];
        end
        
        %optimize and disaggregation
        eccost = spotprices(1:T) * transpose(F1T.getEnergyVars());
        time0 = tic;
        eSolution = optimize(ectr, eccost,sett);
        timeOptMinkowski = toc(time0);
        MinkowskiCost = value(eccost);
        EnergyValuesM = value(F1T.getEnergyVars);
        
        time0 = tic;
        DisaggEnergyM = disaggregate(F1T,LTImodels);
        timeMDisaggregate = toc(time0);
    end
    
    %Calculate profit - Approximate Minkowski
    
    if UseAppMink == 1
        %convert polyhedron to constraints
        Representation = ApproxMinkowski.H; 
        ectr = []; 
        for k = 1:length(Representation)
            Rep = Representation(k,:);
            ectr = [ectr, Rep(1:T)*F1.getEnergyVars()' <= Rep(T+1)];
        end
        
        %optimize
        eccost = spotprices(1:T) * transpose(F1T.getEnergyVars());
        time0 = tic;
        eSolution = optimize(ectr, eccost,sett);
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
    end
    
    %Calculate profit - DFO
    
    if UseDFO == 1
        %optimize
        DFOcost = spotprices(1:T) * transpose(AggregateDFO.getEnergyVars());
        time0 = tic;
        DFOSolution = optimize(AggregateDFO.getAllConstraints(),DFOcost,sett);
        timeOptDFO = toc(time0);
        DFOcostValue = value(DFOcost);
        DFOEnergyValue = value(AggregateDFO.getEnergyVars);
        
        %imbalance calculation and disaggregation
%         constraints = F1T.getAllConstraints;
%         Imbalance = imbalance_calculation(F1T.getEnergyVars(),DFOEnergyValue,imbalanceprices(1:T),constraints);
%         DFOEnergyImbalance = Imbalance(1);
%         DFOCostImbalance = Imbalance(2);
%         DFOTotalCost = DFOcostValue + DFOCostImbalance;
        
        %time0 = tic;
        %d_err = disaggregate(AggregateDFO, DFOmodels);
        %timeDFODisaggregate = toc(time0);
    end
    if UseUFO == 1
    for pt2 = 1:1
        pt1 = 110-10*pt2;
        pt = (pt1/100)^NumBatteries;
    % Calculate profit - ProbFO
    
    PFO1 = DFOSystem(data1,T,pt,"prob");
    PFO2 = DFOSystem(data2,T,pt,"prob");
    PFOmodels = repelem([PFO1,PFO2],NumBatteries/2);
    time0 = tic;
    PFO = DFOSystem(PFOmodels);
    timenewagg = toc(time0);
    PFOcost = spotprices(1:T) * transpose(PFO.getEnergyVars());
    PFOSolution = optimize(PFO.getAllConstraints,PFOcost,sett);
    PFOcostValue = value(PFOcost);
    PFOEnergyValue = value(PFO.getEnergyVars);
    PFOEnergyValue1 = repelem(0,T);
    PFOEnergyValue1(1) = DFOEnergyValue(1);
    time0 = tic;
    d_err = disaggregate(PFO, PFOmodels);
    timePFODisagg = toc(time0);
%     time0 = tic;
%     PFOEnergyValuee = DPFOCtrAppAgg(DFOmodels,data,DFOEnergyValue,spotprices,imbalanceprices,pt,sett);
%     PFOEnergyValue = PFOEnergyValuee.EV;
%     timenewagg = toc(time0);
%     PFOcostValue = value(spotprices*PFOEnergyValue');
% 
%     time0 = tic;
%     PFODisagg = DPFODisagg(data,PFOEnergyValuee,pt);
%     timePFODisagg = toc(time0);
    
    %imbalance calculation and disaggregation
%     constraints = F1T.getAllConstraints;
%     Imbalance = imbalance_calculation(F1T.getEnergyVars(),PFOEnergyValue,imbalanceprices(1:T),constraints);
%     PFOEnergyImbalance = Imbalance(1);
%     PFOCostImbalance = Imbalance(2);
%     PFOTotalCost = PFOcostValue + PFOCostImbalance;
    
    end
    end
    
    if UseUncBase == 1
        DFOU1 = DFOSystem(F1,0.4,data1);
    DFOU2 = DFOSystem(F1,0.4,data2);
    time0 = tic;
    DFOUmodels = repmat([DFOU1,DFOU2],1,NumBatteries/2);
    DFOU = DFOSystem(DFOUmodels);
    timeUncBaseAgg = toc(time0);
    AggregateDFO = DFOUmodels(1).slices;
    ODFO = DFOSystem(DFOUmodels(1).slices);
    time0 = tic;
    for k = 1 : NumBatteries - 1 %aggregate one DFO at a time
        [~,~,AggregateDFO] = aggregate_slices_determ3(ODFO,AggregateDFO,DFOUmodels(k+1).slices,NumSamples);
%        ODFO = DFOSystem(AggregateDFO,F1.getEnergyVars()); %convert it back to a DFO system
    end
    DFO = DFOSystem(AggregateDFO,F1.getEnergyVars());
    timeAggregateDFO = toc(time0);
    DFOUcost = spotprices(1:T) * transpose(DFOU.getEnergyVars());
    
    %optimize
    time0 = tic;
    DFOUSolution = optimize(DFOU.getAllConstraints(),DFOUcost,sett);
    timeOptUncBase = toc(time0);
    DFOUcostValue = value(DFOUcost);
    DFOUEnergyValue = value(DFOU.getEnergyVars);
    %%FROM HERE TO DISAGGREGATE IS FOR THE OPT TIME TEST
    RT = [RT,timeOptUncBase];
    end
end

%     % disaggregate
%     time0 = tic;
%     d_err = disaggregate(DFOU, DFOUmodels);
%     timeUBDisaggregate = toc(time0);
%     end
%     
%     % DETERMINE SoC
%     
%     Q = repelem(0,NumBatteries,T+1);
%     Q(:,1) = Q0;
%     for t = 1:T
%         for k = 1:NumBatteries
%             Energy = LTImodels(k).getEnergyValues;
%             Q(k,t+1) = decay(k)*Q(k,t) + L(k)*max(Energy(t),0) + (1/L(k))*min(Energy(t),0);
%         end
%     end
%     
%     for k = 1:NumBatteries
%         SoC0(k,counter+1) = max(0,min(Qmax(k),Q(k,T+1)));
%     end
% 
%     %%%%% ADD RESULTS
%     
%     if UseDFO == 1
%         RMatrix(1,counter) = DFOAggTime;
%         RMatrix(2,counter) = timeOptDFO;
%         RMatrix(3,counter) = timenewagg;
%         RMatrix(4,counter) = timePFODisagg;
%     elseif UseNoAgg == 1 && UseAppMink == 0 && UseMinkowski == 0
%         RMatrix(1,counter) = timeBigModel;
%         RMatrix(2,counter) = timeOptLTIUnion;
%         RMatrix(3,counter) = timeNoAggDisaggregate;
%     elseif UseAppMink == 1
%         RMatrix(1,counter) = timeAppMinkAgg;
%         RMatrix(2,counter) = timeOptAMinkowski;
%         RMatrix(3,counter) = timeAMDisaggregate;
%     elseif UseMinkowski == 1
%         RMatrix(1,counter) = timeMinkowskiAgg;
%         RMatrix(2,counter) = timeOptMinkowski;
%         RMatrix(3,counter) = timeMDisaggregate;
%     elseif UseUncBase == 1
%         RMatrix(1,counter) = timeUncBaseAgg;
%         RMatrix(2,counter) = timeOptUncBase;
%         RMatrix(3,counter) = timeUBDisaggregate;
%         RMatrix(4,counter) = timeAggregateDFO;
%    end
%     RT = [RT;RMatrix(1:4,1)'];
%     disp('Counter is');
%     disp(value(counter));
%     disp('Parallel is');
%     disp(value(parallel));
% end
%     %filename = sprintf('/home/ubuntu/imbalance_Q0is0_Tis6_Typeis0_03_daysare1to372part%d.xlsx',parallel);
% %     if cloud == 1
% %         filename = sprintf("/home/ubuntu/DFOAswitching%d_%d_%d_%d_%d.xlsx",T,NumBatteries,dif,parallel,pt1);
% %     else
% %         filename = "C:\Users\Fab\Documents\Postdoc\Disaggregate"+T+"_"+NumBatteries+".xlsx";
% %     end
% %     writematrix(RMatrix,filename)
% 
 end
end
end
RR = reshape(RT,[11,8])';
R = repelem(0,8);
Re = sort(RR,2);
for k = 1:8
    R(k) = sum(Re(k,4:8))/5;
end