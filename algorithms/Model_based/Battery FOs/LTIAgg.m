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


for parallel = 3:15 %change to "for" or "parfor" 
for NumBatteries = 20:10:200
T = parallel; %generation/planning horizon 
%NumBatteries = 100; %number of batteries; better to keep it even
    
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
    if bignumbers == 1
        spos = T*(counter - 1);
    else
        spos = T*(counter - 1) + T*Length*(parallel-1); %for adjusting time depending on
    end                                       %the "counter" and "parallel" values
                                               
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
    timerzero = tic;
%    ODFO1 = DFOSystem(F1,0.1);
%    ODFO2 = DFOSystem(F2,0.2);
    disp("DFO");
    toc(timerzero)
    timerzero = tic;
%     F1Poly = F1.getEnergyPolyhedron;
%     F2Poly = F2.getEnergyPolyhedron;
%     LTImodels = repmat([F1,F2],1,NumBatteries/2);
%     LTIpolys = repmat([F1Poly,F2Poly],NumBatteries/2,1);
%     DFOmodels = repmat([ODFO1,ODFO2],1,NumBatteries/2);
    disp("Create the vectors");
    toc(timerzero)
    
   
        timerzero = tic;
        A = eye(NumBatteries);
        B3 = repelem(0,NumBatteries,2*NumBatteries);
        for k = 1 : NumBatteries
            B3(k,2*k-1) = L(k);
            B3(k,2*k) = 1/L(k);
        end
        f = repelem(0,NumBatteries,NumBatteries);
        C = repelem(0,1,NumBatteries);
        D1 = repelem(1,2*NumBatteries);
%         D1 = repelem(0,NumBatteries,2*NumBatteries);
%         for k = 1 : NumBatteries
%             D1(k,2*k-1) = 1;
%             D1(k,2*k) = -1;
%         end
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
        timeBigModel = toc(timerzero);
    
    %%%%% AGGREGATION
    
    %Aggregation - Minkowski sum
    
%     timerzero = tic;
%     AggregateLTI = LTIpolys(1);
%     Num = size(LTIpolys);
%     time00 = tic;
%     for k = 1 : Num(1)-1
%         AggregateLTI = AggregateLTI + LTIpolys(k+1); %sum all the polyhedra, one by one
%         timebomb = toc(time00); %abort after a certain time
%         if timebomb > limit
%             disp(k);
%             break
%         end
%     end
%     timeAggregateMinkovski = toc(time00);
%     disp("Minkowski aggregation")
%     toc(timerzero)
    
    %Aggregation - approximate Minkowski sum
    
%     LTIpolys = num2cell(LTIpolys)'; %convert to cell format
%     time0 = tic;
%     ApproxMinkowski = MK_outer_approx(LTIpolys); %apparently, there was a built-in function for this
%     timeAggregateAM = toc(time0);
%     disp("Approximate Minkowski aggregation");
    
    %Aggregation - DFO
    
%     AggregateDFO = DFOmodels(1).slices;
%     ODFO = DFOSystem(DFOmodels(1).slices);
%     time0 = tic;
%     for k = 1 : NumBatteries - 1 %aggregate one DFO at a time
%         [~,~,AggregateDFO] = aggregate_slices_determ3(ODFO,AggregateDFO,DFOmodels(k+1).slices,NumSamples);
% %        ODFO = DFOSystem(AggregateDFO,F1.getEnergyVars()); %convert it back to a DFO system
%     end
%     ODFO = DFOSystem(AggregateDFO,F1.getEnergyVars());
%     timeAggregateDFO = toc(time0);
%     disp("DFO aggregation")
    
%     DFOcost = spotprices(1:T) * transpose(ODFO.getEnergyVars());
%         time0 = tic;
%         DFOSolution = optimize(ODFO.getAllConstraints(),DFOcost);
%         timeOptDFO = toc(time0);
%         DFOcostValue = value(DFOcost);
%         DFOEnergyValue = value(ODFO.getEnergyVars);
        constraints = F1T.getAllConstraints;
        Q = sdpvar(1,T+1);
        Q(1) = 0;
        eVars = F1T.getEnergyVars();
        for k = 1:T
        Q(k+1) = Q(k) + sum(L)/NumBatteries * max(eVars(k),0) + NumBatteries/sum(L) * min(eVars(k),0);
%   VCost = VCost + Cost*(up(k)+un(k));
        end
        constraints = [constraints, repelem(sum(Pmin),T) <= eVars <= repelem(sum(Pmax),T), repelem(sum(Qmin),T+1) <= Q <= repelem(sum(Qmax),T+1)];
        %time0 = tic;
        %costbattery = spotprices(1:T) * F1T.getEnergyVars()';
        %optimize(constraints,costbattery);
        %timeOptLTIUnion = toc(time0);
        cLTI = ltiT.constraints;
        LTIcost = spotprices(1:T) * ltiT.y.var';
        optimize(cLTI,LTIcost);
        aggrcost = value(costbattery);
        time0 = tic;
        value(ltiT.x.var);
        timeLTI = toc(time0);
        EnergyValuesUnion = value(F1T.getEnergyVars); %Save the energy values for calculating Q

    
    %%%%% ADD RESULTS
    
    RMatrix(1,counter) = T;%timeBigModel;
    RMatrix(2,counter) = NumBatteries;%timeOptLTIUnion;
    RMatrix(3,counter) = timeLTI;
%     RMatrix(2,counter) = timeAggregateDFO;
%     RMatrix(1,counter) = timeOptDFO;
    disp('Counter is');
    disp(value(counter));
    disp('Parallel is');
    disp(value(parallel));
end
    %filename = sprintf('/home/ubuntu/imbalance_Q0is0_Tis6_Typeis0_03_daysare1to372part%d.xlsx',parallel);
    if cloud == 1
        filename = sprintf("/home/ubuntu/AllThree%d_%dNumSamples%d.xlsx",T,NumBatteries,NumSamples);
    else
        filename = "C:\Users\Fab\Documents\Postdoc\LTIDisagg"+T+"_"+NumBatteries+".xlsx";
    end
    writematrix(RMatrix,filename)

end
end