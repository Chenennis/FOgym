function Imbalance = imbalance_calculation(eVars,nVars,prices,cCtr,sett)

if nargin < 5
    sett = sdpsettings('solver','sedumi','verbose',0);
end
%%% IMBALANCE CALCULATION %%%

% Get the approximate energy schedule 
approxSchedule = value(nVars);

% TEMPORARY - TO BE EVENTUALLY DELETED
siz = size(prices);
prices = max(prices, repelem(0,siz(2)));

% Compute the imbalances
imbalanceAmount = abs(eVars - approxSchedule);
imbalanceCost = sum(prices .* imbalanceAmount);
optimize(cCtr, imbalanceCost,sett);


% Extract results
totalImbalanceAmount = sum(value(imbalanceAmount)); 
totalImbalanceCost = sum(value(imbalanceCost));
Imbalance = [totalImbalanceAmount,totalImbalanceCost];
end