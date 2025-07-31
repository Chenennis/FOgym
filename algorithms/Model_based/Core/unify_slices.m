function slicesU = unify_slices( obj, slicesM, slicesS)
% This method unifies slices, so that no flexibility will be lost on the 
% "master" slices, after aggregating it with "slave" slices

N = max(length(slicesM), length(slicesS));

if length(slicesM) ~= length(slicesS)
    warning('Aggregating DFOSystems with different number of slices! Adding extra slices');
    
    % Add additional slices to Nr. 1
    if ~isempty(slicesM)
       [tmin1, tmax1] = obj.getEnergyTotalMinMax(slicesM(length(slicesM)));
    else
       tmin1 = 0;
       tmax1 = 0;
    end
    
    for t = length(slicesM) + 1 : N        
        slicesM = [slicesM, Polyhedron('lb', [tmin1, 0], 'ub', [tmax1 0]).minHRep().normalize()];
    end
    
    % Add additional slices to Nr. 2
    if ~isempty(slicesS)
       [tmin2, tmax2] = obj.getEnergyTotalMinMax(slicesS(length(slicesS)));
    else
       tmin2 = 0;
       tmax2 = 0;
    end
    
    for t = length(slicesS) + 1 : N        
        slicesS = [slicesS, Polyhedron('lb', [tmin1, 0], 'ub', [tmax1 0]).minHRep().normalize()];
    end    
end

ops = sdpsettings('verbose',0);

% Offset
ve = sdpvar(1, N);
% Scale
vs = sdpvar(1, 1);

%fprintf('u.');
     
C = [];
for t = 1:N
   
 % Inscribe slice M into slice S
 Vm = slicesM(t).V;
 
 As = slicesS(t).A;
 bs = slicesS(t).b;
 
 % Must fit into the slave's constraints
 for i = 1 : size(Vm, 1)
    if t == 1
        C = [C;  As * [ 0; vs * Vm(i, 2) + ve(t)] <= bs ];
    else
        C = [C;  As * [ vs * Vm(i, 1) + sum(ve(1:(t-1))); vs * Vm(i, 2) + ve(t)] <= bs ];
    end;
 end

end;

 % Maximize the scale
sol= optimize(C, -vs, ops); 

if sol.problem ~= 0
    warning('Failed finding the scaling of the master slice!');
    slicesU = [];
    return;
end
  
% Rebuild slices
slicesU = [];
for t = 1:N
     % Compute intersection
     if t == 1
         P = slicesM(t) * value(vs) + [ 0; value(ve(t))];
     else
         P = slicesM(t) * value(vs) + [ sum(value(ve(1:t-1))); value(ve(t))];
     end
     slicesU = [slicesU; P.minVRep().minHRep().normalize()]; 
end

end

