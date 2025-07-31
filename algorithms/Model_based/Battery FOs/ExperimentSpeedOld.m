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


for T  = 24 : 8 : 96 %change to "for" or "parfor" 
T = 12;
%T = 3; %generation/planning horizon 
%NumBatteries = parallel; %number of batteries; better to keep it even
    
%Parameters of the algorithm
NumSamples = 3 ; %if needed, type of FO/DFO used
limit = 1800 ; %number of seconds before aborting a process
bignumbers = 1; %change to 1 when processing 2000+ batteries

%Parameters of the battery
InitialSoC = 7; %initial state of charge
L = sqrt(0.9); %loss of the battery (square root of roundtrip efficiency)
Qmin = 0; %minimum possible state of charge
Qmax = 14;%maximum possible state of charge
Pmin = -5; %maximum discharging power
Pmax = 5; %maximum charging power
decay = 1; %decay at each time unit
percentage = 100; %percentage of flexibility exploited by DFO


    
if visual == 1
   T = 3; %set 3D if visualization is wanted 
end

days = 3;%number of days for iteration
Length = 5;%floor(days*24/T); %number of cycles needed
if testing == 1
    Length = 1;
end


RMatrix = repelem(0,2,Length); %initializing result matrix
SoC0 = repelem(0,Length+1); %initializing "continuity array", 
          %it describes the state of charge at the beginning of next cycle
SoC0(1) = InitialSoC'; %initial state of charge
daysahead = 48; %how many days ahead we want to start; mainly for debugging
for counter = 1:Length
    timerzero = tic;
    Q0 = 14*rand(1); %initial state of charge for the cycle
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
    A = [decay];
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

    P = Polyhedron('lb', [Q0], 'ub', [Q0]);
    lti.x.with('initialSet');
    lti.x.initialSet = P;

    lti.instantiate(T);
    F1 = FlexSystem(lti);
        
    disp("Begin and LTI");
    toc(timerzero)
    
    timerzero = tic;
 %   ODFO = DFOSystem(F1,0);
    disp("DFO");
    toc(timerzero)
    
    target1 = spotprices * F1.getEnergyVars()';
 %   target2 = spotprices * ODFO.getEnergyVars()';
    
    actualtimer = tic;
    optimize(F1.model.constraints,target1);
    timeLTI = toc(actualtimer);
    
%     actualtimer = tic;
%     optimize(ODFO.getAllConstraints,target2);
%     timeDFO = toc(actualtimer);
%     
    uu = value(F1.getEnergyVars);
    
    QQ = sdpvar(1,T+1);
    vv = sdpvar(1,T);
    cc = [QQ(1) == Q0];
    for k = 1 : T
        cc = [cc,QQ(k+1) == QQ(k) + L*max(vv(k),0) + 1/L * min(vv(k),0),Qmin <= QQ(k+1) <= Qmax, L*Pmin <= vv(k) <= Pmax];
    end
    
    target3 = spotprices * vv';
    optimize(cc,target3);
    
    Q = sdpvar(1,T+1);
    Q(1) = Q0;
    for k = 1:T
        Q(k+1) = Q(k) + L*max(uu(k),0) + 1/L * min(uu(k),0);
    end
    
    %%%%% ADD RESULTS
    
    RMatrix(1,counter) = timeLTI;
    RMatrix(2,counter) = timeDFO;
    
    disp('Counter is');
    disp(value(counter));
    disp('Parallel is');
    disp(value(parallel));
end
    %filename = sprintf('/home/ubuntu/imbalance_Q0is0_Tis6_Typeis0_03_daysare1to372part%d.xlsx',parallel);
    if cloud == 1
        filename = sprintf("/home/ubuntu/TimeSingleBattery%d.xlsx",T);
    else
        filename = "C:\Users\Fab\Documents\Postdoc\TimeSingleBattery"+T+".xlsx";
    end
    writematrix(RMatrix,filename)

end