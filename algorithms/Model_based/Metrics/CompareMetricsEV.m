format shortG
yalmip('solver','glpk');
sett = sdpsettings('verbose',0);

%Program definition
cloud = 1; %set to 1 for using it in cloud
par = 16; %number of iterations going in parallel
dif = 50; %threshold for price differences
daystotal = 3; %total days for the simulation
Ttotal = 9; %generation/planning horizon 
sce = 16; %number of scenarios simulated


if cloud == 0
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\M.mat');
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\N.mat');
elseif cloud == 1
    load('/home/ubuntu/MATLABfiles/M.mat');
    load('/home/ubuntu/MATLABfiles/N.mat');
end

hour = 18; %starting hour for charging
timeflex = 5; %time flexibility

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

lti.instantiate(9);
F1 = FlexSystem(lti);

for testcase = 1:1
parfor parallel = 1:par %change to "for" or "parfor" 
    
ActualResults = repelem(0,28,sce); %initializing global result matrix
    for scenarios = 1:sce

    
%Parameters of the battery
InitialSoC = 0; %initial state of charge
L = 0.8+0.08*rand(1); %loss of the battery (square root of roundtrip efficiency)
Qmin0 = 12.5+5*rand(1); %minimum possible state of charge
Qmax = 4*Qmin0;%maximum possible state of charge
Pmin = 0; %maximum discharging power
Pmax = 6+2*rand(1); %maximum charging power
decay = 1; %decay at each time unit
percentage = 100; %percentage of flexibility exploited by DFO

days = ceil(daystotal/par);%number of days for iteration
Length = floor(days); %number of cycles needed

RMatrix = repelem(0,19,Length); %initializing result matrix
daysahead = 0; %how many days ahead we want to start; mainly for debugging

%%Not sure of the purpose, but keep in mind if something fails on cloud

% if par == 8
%     RMatrix(1:7,Length) = repelem(0,7,1);
%     Length = Length - 1;
% end

for counter = 1:Length
    Qmin = Qmin0;
    Q0 = Qmin; %initial state of charge for the cycle
    spos = 24*(counter - 1) + 24*Length*(parallel-1)+hour;
    spotprices = [];
    imbalanceprices = [];
    for k = 1:timeflex
        spotprices = [spotprices;M(k+spos+daysahead:Ttotal+spos+daysahead+k-1,1:1)']; %spot prices
        imbalanceprices = [imbalanceprices;N(k+spos+daysahead:Ttotal+spos+daysahead+k-1,1:1)']; %imbalance prices
    end
    
    data1 = struct();
    data1.InitialSoC = Qmin;
    data1.L = L;
    data1.Pmin = Pmin;
    data1.Pmax = Pmax;
    data1.Qmin = Qmin;
    data1.Qmax = Qmax;
    data1.percentage = 100;
    
    
    ActualEnergy = (Qmax - Q0)/L;
    data1.Emin = ActualEnergy;
    data1.Emax = ActualEnergy;
    F2 = DFOSystem(F1,0.301,data1);
    constraints = [F2.getAllConstraints,sum(F2.getEnergyVars()) == ActualEnergy];
    costs = [];
    maxcosts = [];
    for k = 1:timeflex
    costbattery = spotprices(k,1:Ttotal) * F2.getEnergyVars()';
    optimize(constraints,costbattery,sett);
    costs = [costs;value(costbattery)];
%    EnergyValuesUnion = value(F2.getEnergyVars); %Save the energy values
    %inverse
    profitbattery = -spotprices(k,1:Ttotal) * F2.getEnergyVars()';
    optimize(constraints,profitbattery,sett);
    maxcosts = [maxcosts;-value(profitbattery)];
    end
    realcost = min(costs);
    maxcost = max(maxcosts);
    
    %Compare metrics
    timeflex1 = timeflex;
    
    VectorDFO = vector_flexibility(F2,timeflex1);
    TimeSeriesDFO = time_series_flexibility(F2,timeflex1);
    AssignmentDFO = assignment_flexibility(F2,timeflex1);
    AssignmentDFO2 = assignment_flexibility_granularity(F2,2,timeflex1);
    AssignmentDFO8 = assignment_flexibility_granularity(F2,8,timeflex1);
    AbsoluteAreaDFO = absolute_area_flexibility(F2,timeflex1);
    RelativeAreaDFO = relative_area_flexibility(F2,timeflex1);
    alpha = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500];
    sizealpha = size(alpha);
    le = sizealpha(2);
    Vectoralpha = repelem(0,le);
    for k = 1:le
        Vectoralpha(k) = advanced_vector_flexibility(F2,alpha(k),timeflex1);
    end
    
    %%%%% ADD RESULTS
    
    RMatrix(1,counter) = realcost;
    RMatrix(2,counter) = maxcost;
    RMatrix(3,counter) = VectorDFO;
    RMatrix(4,counter) = TimeSeriesDFO;
    RMatrix(5,counter) = AssignmentDFO;
    RMatrix(6,counter) = AssignmentDFO2;
    RMatrix(7,counter) = AssignmentDFO8;
    RMatrix(8,counter) = AbsoluteAreaDFO;
    RMatrix(9,counter) = RelativeAreaDFO;
    for k = 1:le
        RMatrix(9+k,counter) = Vectoralpha(k);
    end
    
end
%     ActualResults(1,scenarios) = (sum(RMatrix(2,:))-sum(RMatrix(1,:)))/daystotal;
%     ActualResults(2,scenarios) = sum(RMatrix(2,:));
    for k = 1:9+le
        ActualResults(k,scenarios) = sum(RMatrix(k,:))/daystotal;
    end
    ActualResults(10+le,scenarios) = (sum(RMatrix(2,:))-sum(RMatrix(1,:)))/daystotal;

    
    disp('Scenario number');
    disp(value(scenarios));
    end
    %filename = sprintf('/home/ubuntu/imbalance_Q0is0_Tis6_Typeis0_03_daysare1to372part%d.xlsx',parallel);
    if cloud == 1
        filename = sprintf("/home/ubuntu/MetricsEVcorrect%d.xlsx",parallel);
    else
        filename = "C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\MetricsResults\MetricsEV.xlsx";
    end
    writematrix(ActualResults,filename)

end
end