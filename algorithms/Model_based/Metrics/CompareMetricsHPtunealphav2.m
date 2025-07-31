format shortG
yalmip('solver','glpk');
sett = sdpsettings('verbose',0);
warning('off','MATLAB:inpolygon:ModelingWorldLower');

%Program definition
cloud = 1; %set to 1 for using it in cloud
par = 16; %number of iterations going in parallel
time = 900; %length of each time unit, in seconds
daystotal = 112; %number of days that we are considering
T = 12; %generation/planning horizon 
sce = 1; %number of different scenarios simulated

if cloud == 0
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\M.mat');
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\N.mat');
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\Temperatures.mat');
elseif cloud == 1
    load('/home/ubuntu/MATLABfiles/M.mat');
    load('/home/ubuntu/MATLABfiles/N.mat');
    load('/home/ubuntu/MATLABfiles/Temperatures.mat');
end

M1 = repelem(M,3600/time);
N1 = repelem(N,3600/time);
Temperatures = repmat(Temperatures,2,1) + 273.15;

for pt1 = 100
parfor parallel = 1:par %change to "for" or "parfor" 
ActualResults = repelem(0,50,sce); %initializing global result matrix
for scenarios = 1:sce

%Parameters of the HP
% dim1 = 4+2*rand(1);
% dim2 = 3+2*rand(1);
% Area = dim1*dim2; %surface area of the room, in square meters
% Volume = Area*3;
% c_ht = 0.006; %overall heat transfer coefficient multiplied for time, in kW/(m²K)
% c = 1.005; %specific heat capacity of air, in kJ/(kg*K)
% m = 1.225*Volume; %mass of the air inside the room, in kg
% Pmax = 4.6;
% Tlow  = 290+10*rand(1);
% Thigh = Tlow+1+5*rand(1);
% %T_out = Temperatures; % Ambient temp, K
% T_in = (Tlow+Thigh)/2; % Inner starting temperature, K
% COP = 3.5+0.5*rand(1); % coefficient of performance
% k = 1 - exp(-(Area*c_ht*time)/(c*m));
% percentage = 100; %percentage of flexibility exploited by DFO
% pt = pt1/100; %probability threshold
% gr = 100; %granularity

Area = 12; %surface area of the room, in square meters
Volume = 60;
c_ht = 0.002; %overall heat transfer coefficient multiplied for time, in kW/(m²K)
c = 1.005; %specific heat capacity of air, in kJ/(kg*K)
m = 1.225*Volume; %mass of the air inside the room, in kg
Pmax = 4.6;
Tlow = 293;
Thigh = 297;
T_in = 295; % Inner starting temperature, K
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
%data.T_out = T_out;
data.time = time;
data.T = T;
data.percentage = 100;
data2 = data;

days = ceil(daystotal/par);%number of days for iteration
Length = floor(days*24/T*3600/time); %number of cycles needed

RMatrix = repelem(0,19,Length); %initializing partial result matrix
daysahead = 0; %how many days ahead we want to start; mainly for debugging

for counter = 1:Length
   
    spos = T*(counter - 1) + T*Length*(parallel-1);        
    spotprices = M1(1+spos+daysahead:T+spos+daysahead,1:1)'; %spot prices
    imbalanceprices = N1(1+spos+daysahead:T+spos+daysahead,1:1)'; %imbalance prices
    spotpricesCOP = max(spotprices(1:T)/COP,repelem(0.0001,T));
    imbalancepricesCOP = max(imbalanceprices(1:T)/COP,repelem(0.0001,T));
    prices = struct();
    prices.spotprices = spotpricesCOP;
    prices.imbalanceprices = imbalancepricesCOP;
    
    T_out = Temperatures(1+spos+daysahead:T+spos+daysahead,1:1); %import outer temperature
    data.T_out = T_out; %update data on outer temperature
    data2.T_out = T_out(1);
  
    %Model - LTI (just as a dummy)
    
    A = [1 - k];
    B = [ k/(Area*c_ht)];
    f = [ k*T_out(1) ];
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
    
    %Calculate profit - exact constraint
    
    Ttemp = sdpvar(1,T+1);
    Ttemp(1) = T_in;
    ev = sdpvar(1,T);
    for ti = 1:T
        Ttemp(ti+1) = A*Ttemp(ti) + B*ev(ti)*3600/time + k*T_out(ti);
    end
    constraintstruec = [repelem(Tlow,T+1) <= Ttemp <= repelem(Thigh,T+1)];
    cfunc = spotpricesCOP * ev';
    optimize(constraintstruec,cfunc,sett);
    TrueEV = value(ev);
    Truecost = value(cfunc);

    
    
    %Calculate profit - LTI
    
    %optimize
%     constraints = F.getAllConstraints; %Tend == (Tlow+Thigh)/2];
%     costbattery = spotpricesCOP * F.getEnergyVars()';
%     optimize(constraints,costbattery,sett);
%     EnergyValuesUnion = value(F.getEnergyVars)*time/3600; %Save the energy values
%     aggrcost = value(costbattery);
%         
    %Calculate profit - DFO
    
    %create constraints
%    DFO1 = DFOSystem(F,0);
    DFO = DFOSystem(F,0.312,data);
    DFOcost = spotpricesCOP * transpose(DFO.getEnergyVars());
   
    %optimize
    DFOSolution = optimize(DFO.getAllConstraints(),DFOcost,sett);%, TendD == (Tlow+Thigh)/2]
    DFOEnergyValue = value(DFO.getEnergyVars)*time/3600;
    DFOcostValue = value(DFOcost);
    
    %inverse
    DFOcostInverse = -spotpricesCOP * transpose(DFO.getEnergyVars());
    DFOSolutionInverse = optimize(DFO.getAllConstraints(),DFOcostInverse,sett);
    DFOcostI = value(DFOcostInverse);
    
    %imbalance
    DFOImbCost = imbalancepricesCOP * transpose(abs(F.getEnergyVars()-DFOEnergyValue/time*3600));
    DFOSolutionImb = optimize([F.getAllConstraints()],DFOImbCost,sett);
    DFOpenalty = value(DFOImbCost);
    DFOImbEn = value(F.getEnergyVars);
    DFOTotalCostValue = DFOcostValue + DFOpenalty;
    
   %Calculate profit - HFO2
   
    B1 = (Thigh-Tlow)/(energy_opt(data2,Tlow,Thigh)-energy_opt(data2,Tlow,Tlow));
    A1 = B1 * (energy_opt(data2,Tlow,Thigh)-energy_opt(data2,Thigh,Thigh))/(Thigh-Tlow);
    f1 = (1-A1)*Tlow - B*energy_opt(data2,Tlow,Tlow);
    C = [0];
    D = [1];
    lti2 = LTISystem('A', A1, 'B', B1, 'C', C, 'D', D, 'f', f1);
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
    
    %inverse
    HFOcostInverse = -spotpricesCOP * transpose(HFO.getEnergyVars());
    HFOSolutionInverse = optimize(HFO.getAllConstraints(),HFOcostInverse,sett);
    HFOcostI = value(HFOcostInverse);
    
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
        data.currenttime = kk;
        HFOTempCheck(kk+1) = temp_calc_implicit(data,HFOTempCheck(kk),EnergyValuesUnion2(kk)*3600/time);
    end
    
    countchange = repelem(0,T*time/3600);
    for kk1 = 1:T
        ind = floor((kk1-1)/4)+1;
        if abs(HFOTempCheck(kk1+1) - HFOTempCheck(kk1)) > 0.02
            countchange(ind) = countchange(ind)+1;
        end
    end
    
    %candidate perfect 2

    evars = repelem(0,T);
    Tempvar = repelem(0,T+1);
    evars(1) = energy_opt(data2,T_in,Tlow);
    Tempvar (1) = T_in;
    Tempvar (2) = Tlow;
    
    %optimize
    for tt = 2:T
        data.currenttime = tt;
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
        data.currenttime = kk;
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
       Te(t+1) = A*Te(t) + B*TrueEV(t)*3600/time + k*T_out(t);
   end
    T_in = Te(T+1);
    
    %Metrics calculation
    
    %DFO
    
    VectorDFO = vector_flexibility(DFO);
    TimeSeriesDFO = time_series_flexibility(DFO);
    AssignmentDFO = assignment_flexibility(DFO);
    AssignmentDFO2 = assignment_flexibility_granularity(DFO,2);
    AssignmentDFO8 = assignment_flexibility_granularity(DFO,8);
    AbsoluteAreaDFO = absolute_area_flexibility(DFO);
    RelativeAreaDFO = relative_area_flexibility(DFO);
    vresDFO = [value(VectorDFO),value(TimeSeriesDFO),value(AssignmentDFO),value(AssignmentDFO2),value(AssignmentDFO8),value(AbsoluteAreaDFO),value(RelativeAreaDFO)];
    svres = size(vresDFO);
    svres2 = svres(2);
%     alpha = [0.001,0.005,0.1,0.5,1,5,10,50,100,500];
%     salp = size(alpha);
%     salp2 = salp(2);
%     DFOalpha = repelem(0,salp2);
%     for kt = 1:salp2
%         DFOalpha(kt) = advanced_vector_flexibility(DFO,alpha(kt));
%     end
        
    %HFO
    
    VectorHFO = vector_flexibility(HFO);
    TimeSeriesHFO = time_series_flexibility(HFO);
    AssignmentHFO = assignment_flexibility(HFO);
    AssignmentHFO2 = assignment_flexibility_granularity(HFO,2);
    AssignmentHFO8 = assignment_flexibility_granularity(HFO,8);
    AbsoluteAreaHFO = absolute_area_flexibility(HFO);
    RelativeAreaHFO = relative_area_flexibility(HFO);
    vresHFO = [value(VectorHFO),value(TimeSeriesHFO),value(AssignmentHFO),value(AssignmentHFO2),value(AssignmentHFO8),value(AbsoluteAreaHFO),value(RelativeAreaHFO)];
%     HFOalpha = repelem(0,salp2);
%     for kt = 1:salp2
%         HFOalpha(kt) = advanced_vector_flexibility(HFO,alpha(kt));
%     end
    
    %%%%% ADD RESULTS
    
    RMatrix(1,counter) = DFOcostValue/4;
    RMatrix(2,counter) = HFOcostValue/4;
    RMatrix(3,counter) = -DFOcostI/4;
    RMatrix(4,counter) = -HFOcostI/4;
    for ku = 1:svres2
        RMatrix(4+ku,counter) = vresDFO(ku);
        RMatrix(4+ku+svres2,counter) = vresHFO(ku);
    end
%    RMatrix(1:12,counter) = RMatrix(1:12,counter)/4;
%     for kt = 1:salp2
%         RMatrix(2 + 2*svres2 + kt, counter) = DFOalpha(kt);
%         RMatrix(2 + 2*svres2 + salp2 + kt, counter) = HFOalpha(kt);
%     end
    
    parallel1 = num2str(parallel);
    counter1 = num2str(counter);
    tbdsp = ['Parallel ',parallel1,', Counter ',counter1];
    disp(tbdsp);
    
end

%ActualResults(1,scenarios) = sum(RMatrix(4,:))/daystotal;
%ActualResults(9,scenarios) = sum(RMatrix(7,:))/daystotal;

%DE-COMMENT FOR SINGLE SCENARIOS

% ActualResults = RMatrix;

%DE-COMMENT FOR MULTIPLE SCENARIOS
for kv = 1:4+2*svres2
    ActualResults(kv,scenarios) = sum(RMatrix(kv,:))/daystotal;
end
ActualResults(5 + 2*svres2,scenarios) = sum(RMatrix(3,:)-RMatrix(1,:))/daystotal;
ActualResults(6 + 2*svres2,scenarios) = sum(RMatrix(4,:)-RMatrix(2,:))/daystotal;


%     disp('Scenario number');
%     disp(value(scenarios));

end
%filename = sprintf('/home/ubuntu/imbalance_Q0is0_Tis6_Typeis0_03_daysare1to372part%d.xlsx',parallel);
    if cloud == 1
        filename = sprintf("/home/ubuntu/MetricsHPmultiplescenarios%d.xlsx",parallel);
    else
        filename = "C:\Users\Fab\Documents\Postdoc\MetricsHPmultiplescenarios.xlsx";
    end
    writematrix(ActualResults,filename)

end
end
