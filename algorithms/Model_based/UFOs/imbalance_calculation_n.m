function Imbalance = imbalance_calculation(eVars,nVars,prices,cCtr,sett)

if nargin < 5
    sett = sdpsettings('solver','glpk','verbose',0);
end
%%% IMBALANCE CALCULATION %%%

% Get the approximate energy schedule 
approxSchedule = value(nVars);


% Compute the imbalances
imbalanceAmount = abs(eVars - approxSchedule);
imbalanceCost = sum(prices .* imbalanceAmount);
optimize(cCtr, imbalanceCost,sett);


% Extract results
totalImbalanceAmount = sum(value(imbalanceAmount)); 
totalImbalanceCost = sum(value(imbalanceCost));
Imbalance = [totalImbalanceAmount,totalImbalanceCost];
end