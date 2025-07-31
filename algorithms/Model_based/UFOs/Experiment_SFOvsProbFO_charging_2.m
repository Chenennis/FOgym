format shortG
yalmip('solver','sedumi');
sett = sdpsettings('verbose',0);

%Program definition
cloud = 1; %set to 1 for using it in cloud
par = 8; %number of iterations going in parallel
prob = [1 0.99 0.95 0.9 0.85 0.8 0.75 0.7]; %probabilities I want to consider

if cloud == 0
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\M.mat');
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\N.mat');
elseif cloud == 1
    load('/home/ubuntu/MATLABfiles/M.mat');
    load('/home/ubuntu/MATLABfiles/N.mat');
end

for pt = prob
parfor parallel = 1:par %change to "for" or "parfor" 

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
    
    %Calculate profit - LTI model
    
    constraints = F1.getAllConstraints;
    evars = F1.getEnergyVars();
    constraints = [constraints, repelem(0,T/2) <= evars(1:T/2), evars(T/2+1:T) <= repelem(0,T/2), Q0 + L*sum(evars(1:T/2)) + 1/L * sum(evars(T/2+1:T)) >= Qmin];
    costbattery = spotprices(1:T) * evars';
    optimize(constraints,costbattery,sett);
    aggrcost = value(costbattery);
    EnergyValuesUnion = value(F1.getEnergyVars); %Save the energy values for calculating Q
    
    %Calculate profit - SFO-AAO
    
    %create constraints
    evarsSFO = sdpvar(1,T);
    SFOcost = spotprices(1:T) * transpose(evarsSFO);
    c_AAO = const_AAO_charging(data1,TT);
    cAAO = [c_AAO(1,:) <= evarsSFO <= c_AAO(2,:),L*sum(evarsSFO(1:TT))+sum(evarsSFO(TT+1:T))/L >= 0];
    
    %optimize
    SFOSolution = optimize(cAAO,SFOcost,sett);
    SFOcostValue = value(SFOcost);
    SFOEnergyValue = value(evarsSFO);

    %imbalance calculation and disaggregation
    evars = F1.getEnergyVars();
    Imbalance = imbalance_calculation(evars,SFOEnergyValue,imbalanceprices(1:T),constraints);
    SFOEnergyImbalance = Imbalance(1);
    SFOCostImbalance = Imbalance(2);
    SFOTotalCost = SFOcostValue + SFOCostImbalance;
    
    %Calculate profit - SFO-Flat
    
    %create constraints
    evarsSFOF = sdpvar(1,T);
    SFOcostF = spotprices(1:T) * transpose(evarsSFOF);
    c_flat = const_flat_charging(data1,TT);
    cflat = [c_flat(1,:) <= evarsSFOF <= c_flat(2,:), L*sum(evarsSFOF(1:TT))+sum(evarsSFOF(TT+1:T))/L];
    
    %optimize
    SFOSolutionF = optimize(cflat,SFOcostF,sett);
    SFOcostValueF = value(SFOcostF);
    SFOEnergyValueF = value(evarsSFOF);

    %imbalance calculation and disaggregation
    evars = F1.getEnergyVars();
    ImbalanceF = imbalance_calculation(evars,SFOEnergyValueF,imbalanceprices(1:T),constraints);
    SFOEnergyImbalanceF = ImbalanceF(1);
    SFOCostImbalanceF = ImbalanceF(2);
    SFOTotalCostF = SFOcostValueF + SFOCostImbalanceF;
    
    %Calculate profit - ProbFO
    
    %constraints
    evarsPFO = sdpvar(1,T);
    PFOcost = spotprices(1:T) * transpose(evarsPFO);
    probslice1=prob_slices(data1,TT,gr);
    data2 = data1;
    data2.InitialSoC = Qmax;
    probslice2=prob_slices(data2,TT,gr);
    cprob = const_prob_charging(probslice1.slices,probslice2.slices,probslice1.space,pt);
    PC = [cprob(1,:) <= evarsPFO <= cprob(2,:),L*sum(evarsPFO(1:TT))+sum(evarsPFO(TT+1:T))/L];
    
    %optimization
    PFOSolution = optimize(PC,PFOcost,sett);
    PFOcostValue = value(PFOcost);
    PFOEnergyValue = value(evarsPFO);
    PFOEnergyValue1 = repelem(0,T);
    PFOEnergyValue1(1) = PFOEnergyValue(1);
    cont = [1];
    for t = 2:T
        k = sum(PFOEnergyValue1(1:t-1));
        s = PFOEnergyValue(t);
        prob = prob_sum_improved(data1,t,k,s);
        if prob >= pt
            PFOEnergyValue1(t) = PFOEnergyValue(t);
%              [t1,t2] = find_thresholds(data1,t,k,pt);
%              if s > 0.01
%                  PFOEnergyValue(t) = t2;
%              elseif s < -0.01
%                  PFOEnergyValue(t) = t1;
%              else
%                  PFOEnergyValue(t) = 0;
%              end
        else
            [t1,t2] = find_thresholds(data1,t,k,pt);
            if s < 0
                PFOEnergyValue1(t) = t1;
            else
                PFOEnergyValue1(t) = t2;
            end
        end
        if PFOEnergyValue1(t)*PFOEnergyValue1(t-1) >= 0.01 && t <= T-1
            cont = [cont,t];
        else
            if PFOEnergyValue1(t)*PFOEnergyValue1(t-1) >= 0.01 && t == T
                cont = [cont,t];
            end
            sc = size(cont);
            if sc(2) > 1
                v = sdpvar(1,sc(2));
                co = [sum(v) == sum(PFOEnergyValue1(cont))];
                if PFOEnergyValue1(t-1) >= 0.01
                    co = [co, 0 <= v <= Pmax];
                else
                    co = [co, L*Pmin <= v <= 0];
                end
                optimize(co,spotprices(cont)*v',sett);
                PFOEnergyValue1(cont) = value(v);
            end
            cont = [t];
        end
    end
    PFOcostValue1 = value(spotprices(1:T) * PFOEnergyValue1');
    
    %imbalance calculation and disaggregation
    
    %Only first step
    evars = F1.getEnergyVars();
    ImbalanceP = imbalance_calculation(evars,PFOEnergyValue,imbalanceprices(1:T),constraints);
    PFOEnergyImbalance = ImbalanceP(1);
    PFOCostImbalance = ImbalanceP(2);
    PFOTotalCost = PFOcostValue + PFOCostImbalance;
    
    %Second step
    evars = F1.getEnergyVars();
    ImbalanceP1 = imbalance_calculation(evars,PFOEnergyValue1,imbalanceprices(1:T),constraints);
    PFOEnergyImbalance1 = ImbalanceP1(1);
    PFOCostImbalance1 = ImbalanceP1(2);
    PFOTotalCost1 = PFOcostValue1 + PFOCostImbalance1;
    
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
    RMatrix(2,counter) = SFOcostValue;
    RMatrix(3,counter) = SFOCostImbalance;
    RMatrix(4,counter) = SFOTotalCost;     
    RMatrix(5,counter) = SFOcostValueF;
    RMatrix(6,counter) = SFOCostImbalanceF;
    RMatrix(7,counter) = SFOTotalCostF;
    RMatrix(8,counter) = PFOcostValue;
    RMatrix(9,counter) = PFOCostImbalance;
    RMatrix(10,counter) = PFOTotalCost;
    RMatrix(11,counter) = PFOcostValue1;
    RMatrix(12,counter) = PFOCostImbalance1;
    RMatrix(13,counter) = PFOTotalCost1;  
    
    disp('Counter is');
    disp(value(counter));
    disp('Parallel is');
    disp(value(parallel));
end
    %filename = sprintf('/home/ubuntu/imbalance_Q0is0_Tis6_Typeis0_03_daysare1to372part%d.xlsx',parallel);
    if cloud == 1
        filename = sprintf("/home/ubuntu/3SFOChargingZero%d_%d_%d_pt%d.xlsx",T,NumBatteries,parallel,ptt);
    else
        filename = "C:\Users\Fab\Documents\Postdoc\Disaggregate"+T+"_"+NumBatteries+".xlsx";
    end
    writematrix(RMatrix,filename)

end
end