format shortG
yalmip('solver','sedumi');
sett = sdpsettings('verbose',0);
warning('off','all');

%Program definition
cloud = 0; %set to 1 for using it in cloud
par = 8; %number of parallel processes
dif = 50; %threshold for price difference

%choice of approaches
UseNoAgg = 1; %no aggregation approach
UseMinkowski = 0; %Minkowski sum
UseAppMink = 0; %approximate Minkowski sum
UseDFO = 1; %DFO

if cloud == 0
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\M.mat');
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\N.mat');
elseif cloud == 1
    load('/home/ubuntu/MATLABfiles/M.mat');
    load('/home/ubuntu/MATLABfiles/N.mat');
end

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

%only take high penalties
aa = (N-M > dif);
bb = find(aa == 1);
M1 = M(bb);
N1 = N(bb);
%actual experiment

for testcase = 1:1
for NumEVs = 15:5:15
for parallel = 2:par %change to "for" or "parfor" 

T = 14; %generation/planning horizon 
NumEVs = 15; %number of batteries; better to keep it even
    
%Parameters of the algorithm
NumSamples = 3 ; %if needed, type of FO/DFO used
limit = 1800 ; %number of seconds before aborting a process
bignumbers = 0; %change to 1 when processing 2000+ batteries

%Parameters of the battery
InitialSoC = 0; %initial state of charge
L = repmat([0.84],1,NumEVs*4); %loss of the battery (square root of roundtrip efficiency)
Qmin0 = repmat([24,30],1,NumEVs*2); %minimum possible state of charge
Qmax = repmat([48,60],1,NumEVs*2);%maximum possible state of charge
Pmin = repmat([0],1,NumEVs*4); %maximum discharging power
Pmax = repmat([7],1,NumEVs*4); %maximum charging power
decay = repelem(1,NumEVs*4); %decay at each time unit
percentage = 100; %percentage of flexibility exploited by DFO
pt = 0.9; %probability threshold

probability1 = [0.5 0.876 0.966 1];
probability2 = [1 0.92 0.5];
s1 = size(probability1);
s2 = size(probability2);
Ttotal = T + s1(2) + s2(2);
% t1 = min(find(probability1 > pt))-1;
% t2 = min(find(probability2 > 1-pt))-1;
hourstart = [repelem(16,NumEVs),repelem(17,NumEVs),repelem(18,NumEVs),repelem(19,NumEVs)]; %at which hour does the charging start
hourfinish = [repelem(19+T,2*NumEVs),repelem(20+T,2*NumEVs)]; 
EE = 95;
EUnc = EE/100; %existence probability weekdays
EW = 20;
EWnc = EW/100; %existence probability weekend

%matrix determining probability for each EV to be active at time T
Probmatrix = [repmat([probability1, repelem(1,T-1),probability2,0],NumEVs,1);repmat([0,probability1, repelem(1,T-2),probability2,0],NumEVs,1);repmat([0,0,probability1, repelem(1,T-2),probability2],NumEVs,1);repmat([0,0,0,probability1, repelem(1,T-3),probability2],NumEVs,1)];

days = floor(119/par);%number of days for iteration
Length = floor(days*24/Ttotal); %number of cycles needed

RMatrix = repelem(0,36,Length); %initializing result matrix
SoC0 = repelem(0,NumEVs,Length+1); %initializing "continuity array", 
          %it describes the state of charge at the beginning of next cycle
SoC0(:,1) = InitialSoC'; %initial state of charge
daysahead = 0; %how many days ahead we want to start; mainly for debugging

    
for counter = 1:Length
    
    %isolate high penalties
    if bignumbers == 1 
        spos = Ttotal*(counter - 1);%for adjusting time depending on                        %the "counter" and "parallel" values
    else                       %the "counter" and "parallel" values
        spos = Ttotal*(counter - 1) + Ttotal*Length*(parallel-1);        
    end
    spotprices = M1(1+spos+daysahead:Ttotal+spos+daysahead,1:1)'; %spot prices
    imbalanceprices = N1(1+spos+daysahead:Ttotal+spos+daysahead,1:1)'; %imbalance prices
    
    %instantiate case
    Qmin = Qmin0*(5/6+1/3*pt);
    Q0 = Qmin0 + (Qmin0/10).*normrnd(0,1.5,1,4*NumEVs); %initial state of charge for the cycle
    instantiatebegin = rand(1,4*NumEVs);
    instantiateend = rand(1,4*NumEVs);
    instantiateexist = rand(1,4*NumEVs);
    corrfactor = 1;%(25)/(60-25-10*pt);    

    if mod(counter,7) > 4
        exist = (instantiateexist < EWnc);
        Probmatrixday = EWnc * Probmatrix;
        ProbmatrixB = Probmatrixday;
        baselinecharge = EWnc * sum(Qmax-Qmin);
        UFOcharge = EWnc * corrfactor * sum(Qmax-Qmin);
    else
        exist = (instantiateexist < EUnc);
        Probmatrixday = EUnc * Probmatrix;
        ProbmatrixB = Probmatrixday;
        baselinecharge = EUnc * sum(Qmax-Qmin);
        UFOcharge = corrfactor * sum(Qmax-Qmin);
    end
    i1 = repelem(0,4*NumEVs);
    i2 = repelem(0,4*NumEVs);
    for k = 1:4*NumEVs
        i1(k) = min(find(probability1 > instantiatebegin(k)))-1;
        i2(k) = max(find(probability2 > instantiateend(k)))-1;
    end
    totalcharge = exist*(Qmax-Q0)';
    
    Eventmatrix = repelem(0,4*NumEVs,Ttotal);
    for k = 1:NumEVs
        Eventmatrix(k,:) = exist(k)*[repelem(0,i1(k)), repelem(1,Ttotal-1-i1(k)-i2(k)),repelem(0,i2(k)+1)];
        Eventmatrix(NumEVs+k,:) = exist(NumEVs+k)*[repelem(0,i1(NumEVs+k)+1), repelem(1,Ttotal-2-i1(NumEVs+k)-i2(NumEVs+k)),repelem(0,i2(NumEVs+k)+1)];
        Eventmatrix(2*NumEVs+k,:) = exist(2*NumEVs+k)*[repelem(0,i1(2*NumEVs+k)+2), repelem(1,Ttotal-2-i1(2*NumEVs+k)-i2(2*NumEVs+k)),repelem(0,i2(2*NumEVs+k))];
        Eventmatrix(3*NumEVs+k,:) = exist(3*NumEVs+k)*[repelem(0,i1(3*NumEVs+k)+3), repelem(1,Ttotal-3-i1(3*NumEVs+k)-i2(3*NumEVs+k)),repelem(0,i2(3*NumEVs+k))];
    end

    %%%%% Instantiate models for each battery
    
    UncertainBFOs = [];
    EventFOs = [];
    UncertainFOs = [];
    for k = 1:4*NumEVs
        data1 = struct();
        data1.InitialSoC = Q0(k);
        data1.L = L(k);
        data1.Pmin = Pmin(k);
        data1.Pmax = Pmax(k);
        data1.Qmin = Q0(k);
        data1.Qmax = Qmax(k);
        data1.percentage = percentage;
        data2 = data1;
        data3 = data1;
        data1.partial = ProbmatrixB(k,:);
        data1.InitialSoC = Qmin(k);
        data1.Qmin = Qmin(k);
        data2.partial = Eventmatrix(k,:);
        data3.partial = Probmatrixday(k,:);
        data3.InitialSoC = Qmin(k);
        data3.Qmin = Qmin(k);
        UncertainBFOs = [UncertainBFOs,DFOSystem(F1,0.311,data1)];
        EventFOs = [EventFOs,DFOSystem(F1,0.311,data2)];
        UncertainFOs = [UncertainFOs,DFOSystem(F1,0.311,data3)];
    end
        
    %instantiate models
        
    %%%%% AGGREGATION
        
    %Aggregation
    
    AggregateUFO = DFOSystem(UncertainFOs);
    AggregateExact = DFOSystem(EventFOs);
    AggregateBFO = DFOSystem(UncertainBFOs);
    lastUFO = AggregateUFO.slices(Ttotal);
    lastExact = AggregateExact.slices(Ttotal);
    lastBFO = AggregateBFO.slices(Ttotal);
    totalUFO1 = max(lastUFO.V(:,1));
    totalExact1 = max(lastExact.V(:,1));
    totalBFO1 = max(lastBFO.V(:,1));
    totalUFO = min(totalUFO1,UFOcharge);
    totalExact = min(totalExact1,totalcharge);
    totalBFO = min(totalBFO1,baselinecharge);
MaxExact = repelem(0,Ttotal);
for k = 1:Ttotal
slice = AggregateExact.slices(k);
MaxExact(k) = max(slice.V(:,2));
end
%     AggregateUFO = UncertainFOs(1).slices;
%     UFO = UncertainFOs(1);
%     for k = 1 : 4*NumEVs - 1 %aggregate one DFO at a time
%         [~,~,AggregateUFO] = aggregate_slices_determ3(UFO,AggregateUFO,UncertainFOs(k+1).slices,20);
%         UFO = DFOSystem(AggregateUFO,UncertainFOs(1).getEnergyVars()); %convert it back to a DFO system
%     end
  
    %Calculate profit
    
    exactvars = AggregateExact.getEnergyVars();
    UFOvars = AggregateUFO.getEnergyVars();
    BFOvars = AggregateBFO.getEnergyVars();
    exactcost = spotprices(1:Ttotal) * exactvars';
    UFOcost = spotprices(1:Ttotal) * UFOvars';
    BFOcost = spotprices(1:Ttotal) * BFOvars';
        
    %optimize
    exactsolution = optimize([0 <= exactvars <= MaxExact,sum(exactvars) == totalExact],exactcost,sett);
    exactcostValue = value(exactcost);
    exactEnergyValue = value(exactvars);
    
    UFOsolution = optimize([AggregateUFO.getAllConstraints(),sum(UFOvars) == totalUFO],UFOcost,sett);
    UFOcostValue = value(UFOcost);
    UFOEnergyValue = value(UFOvars);    
    if max(UFOEnergyValue) < 1
        UFOcorrect = optimize(AggregateUFO.getAllConstraints(),-sum(AggregateUFO.getEnergyVars),sett);
        UFOcostValue = value(UFOcost);
        UFOEnergyValue = value(UFOvars);         
    end

    BFOsolution = optimize([AggregateBFO.getAllConstraints(),sum(BFOvars) == totalBFO],BFOcost,sett);
    BFOcostValue = value(BFOcost);
    BFOEnergyValue = value(BFOvars);    
    if max(BFOEnergyValue) < 1
        BFOcorrect = optimize(AggregateBFO.getAllConstraints(),-sum(AggregateBFO.getEnergyVars),sett);
        BFOcostValue = value(BFOcost);
        BFOEnergyValue = value(BFOvars);         
    end
        
        %imbalance calculation and disaggregation
        IVars = AggregateExact.getEnergyVars();
        ImbTariff = imbalanceprices(1:Ttotal);% - spotprices(1:Ttotal);
        ImbalanceF = ImbTariff*IVars';
        MV = max(MaxExact-UFOEnergyValue,0);
        optimbalance = optimize([0 <= IVars <= MV,sum(IVars) == totalExact-totalUFO],ImbalanceF,sett);
        Imbalance = value(IVars)+max(UFOEnergyValue-MaxExact,0);
        ImbalanceCost = ImbTariff*Imbalance';
        UFOTotalCost = UFOcostValue + ImbalanceCost;
        
        IVarsB = AggregateExact.getEnergyVars();
        ImbTariff = imbalanceprices(1:Ttotal); %- spotprices(1:Ttotal);
        ImbalanceBF = ImbTariff*IVarsB';
        MV = max(MaxExact-BFOEnergyValue,0);
        optimbalanceB = optimize([0 <= IVarsB <= MV,sum(IVarsB) == totalExact-totalBFO],ImbalanceBF,sett);
        ImbalanceB = value(IVars)++max(BFOEnergyValue-MaxExact,0);
        ImbalanceBCost = ImbTariff*ImbalanceB';
        BFOTotalCost = BFOcostValue + ImbalanceBCost;

    %%%%% ADD RESULTS
    
    RMatrix(1,counter) = exactcostValue;
    RMatrix(2,counter) = BFOcostValue;
    RMatrix(3,counter) = ImbalanceBCost;
    RMatrix(4,counter) = BFOTotalCost;
    RMatrix(5,counter) = UFOcostValue;
    RMatrix(6,counter) = ImbalanceCost;
    RMatrix(7,counter) = UFOTotalCost;
    
    disp('Counter is');
    disp(value(counter));
    disp('Parallel is');
    disp(value(parallel));
end
    %filename = sprintf('/home/ubuntu/imbalance_Q0is0_Tis6_Typeis0_03_daysare1to372part%d.xlsx',parallel);
    if cloud == 1
        filename = sprintf("/home/ubuntu/AggBEV%d_%d_%dT_%d.xlsx",4*NumEVs,dif,parallel,testcase);
    else
        filename = "C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\AggBEV"+T+"_"+dif+"_"+parallel+"T_"+testcase+".xlsx";
    end
    writematrix(RMatrix,filename)

end
end
end