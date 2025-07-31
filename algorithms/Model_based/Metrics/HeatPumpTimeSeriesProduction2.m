%Version 1: T_out, T_low and T_high are input. 
%Temperatures are expressed in Kelvin

format shortG
yalmip('solver','glpk');
sett = sdpsettings('verbose',0);

%Program definition
cloud = 0; %set to 1 for using it in cloud
par = 1; %number of iterations going in parallel
time = 900; %length of each time unit, in seconds
daystotal = 30; %number of days that we are considering
T = 4; %generation/planning horizon 
timerunning = datetime(2018,01,01,0,0,0,0); %initial time

if cloud == 0
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\M.mat');
    load('C:\Users\IQ24VR\OneDrive - Aalborg Universitet\Skrivebord\Postdoc\MATLAB files\N.mat');
elseif cloud == 1
    load('/home/ubuntu/MATLABfiles/M.mat');
    load('/home/ubuntu/MATLABfiles/N.mat');
end

M1 = repelem(M,3600/time);
N1 = repelem(N,3600/time);
%T_low = repelem(298,25896*3600/time)'; %to be replaced by input - needs to be a vector
%T_high = repelem(300,25896*3600/time)'; %to be replaced by input - needs to be a vector
%T_out = repelem(280,25896*3600/time)'; %to be replaced by input - needs to be a vector
T_low = repmat([298:0.05:300.35,300.3:-0.05:298.05],1,25896/96*3600/time)';
T_high = repmat([302:0.05:304.35,304.3:-0.05:302.05],1,25896/96*3600/time)';
T_out = repmat([280:0.25:292,291.75:-0.25:280.25],1,25896/96*3600/time)';
Output = [];
Outputtime = [];

for testcase = 1:1
for pt1 = 100
for parallel = 1:par %change to "for" or "parfor" 

%Parameters of the HP
Area = 12; %surface area of the room, in square meters
Volume = 60;
c_ht = 0.006; %overall heat transfer coefficient multiplied for time, in kW/(m²K)
c = 1.005; %specific heat capacity of air, in kJ/(kg*K)
m = 1.225*Volume; %mass of the air inside the room, in kg
Pmax = 4.6;
T_in = 300; % Inner starting temperature, K
COP = 3.65; % coefficient of performance
k = 1 - exp(-(Area*c_ht*time)/(c*m));
percentage = 100; %percentage of flexibility exploited by DFO
pt = pt1/100; %probability threshold
gr = 100; %granularity

days = ceil(daystotal/par);%number of days for iteration
Length = floor(days*24/T*3600/time); %number of cycles needed

RMatrix = repelem(0,19,Length); %initializing result matrix
daysahead = 0; %how many days ahead we want to start; mainly for debugging

for counter = 1:Length
   
    spos = T*(counter - 1) + T*Length*(parallel-1);        
    spotprices = M1(1+spos+daysahead:T+spos+daysahead,1:1)'; %spot prices
    imbalanceprices = N1(1+spos+daysahead:T+spos+daysahead,1:1)'; %imbalance prices
    T_lowV = T_low(1+spos+daysahead:T+spos+daysahead+1,1:1)'; %minimum constraint on temperature
    T_highV = T_high(1+spos+daysahead:T+spos+daysahead+1,1:1)'; %maximum constraint on temperature
    T_outV = T_out(1+spos+daysahead:T+spos+daysahead,1:1)'; %outer temperature
    spotpricesCOP = max(spotprices(1:T)/COP,repelem(0.0001,T));
    imbalancepricesCOP = max(imbalanceprices(1:T)/COP,repelem(0.0001,T));
    prices = struct();
    prices.spotprices = spotpricesCOP;
    prices.imbalanceprices = imbalancepricesCOP;
    
    %LTI parameters setting
    A = [1 - k];
    B = [ k/(Area*c_ht)];
    
            %Calculate profit - exact constraint

    Ttemp = sdpvar(1,T+1);
    Ttemp(1) = T_in;
    ev = sdpvar(1,T);
    for ti = 1:T
        Ttemp(ti+1) = A*Ttemp(ti) + B*ev(ti)*3600/time + k*T_outV(ti);
    end
    constraintstruec = [T_lowV <= Ttemp <= T_highV];
    cfunc = spotpricesCOP * ev';
    optimize(constraintstruec,cfunc,sett);
    Energy = value(ev);
    Truecost = value(cfunc);
    T_inV = value(Ttemp);

   %Define energy variable
   
    %%%%% ADD RESULTS
    
    
    for t = 1:T
        ov = [T_inV(t),T_outV(t),Energy(t),T_lowV(t),T_highV(t)];
        Output = [Output;ov];
        Outputtime = [Outputtime;timerunning];
        timerunning.Minute = timerunning.Minute + 15;
    end
    T_in = T_inV(T+1);
    
    disp('Counter is');
    disp(value(counter));
    disp('Parallel is');
    disp(value(parallel));
end
    %filename = sprintf('/home/ubuntu/imbalance_Q0is0_Tis6_Typeis0_03_daysare1to372part%d.xlsx',parallel);
    if cloud == 1
        filename = sprintf("/home/ubuntu/HeatHFOsDualLTIcountlong%d_%d.xlsx",parallel,T);
    else
        filename1 = "C:\Users\Fab\Documents\Postdoc\HeatPumpDataPoints.xlsx";
        filename2 = "C:\Users\Fab\Documents\Postdoc\HeatPumpTimestamps.xlsx";
    end
    writematrix(Output,filename1)
    writematrix(Outputtime,filename2)

end
end
end