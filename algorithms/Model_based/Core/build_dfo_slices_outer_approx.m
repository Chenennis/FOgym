function slices = build_dfo_slices_outer_approx(obj, mCtr, eVars, NumSamples)
 T = size(eVars, 2);
 disp(value(T));
 
 % Add additional constraints
 cCtr = mCtr;
 eTotal = sdpvar(1, T);
  
 %for t = 1:T
 %    cCtr = [cCtr; eTotal(t) == sum(eVars(1:t))];
 %end

% Display message
fprintf('Building DFO slices.');

slices = [];
for t = 1:T
    fprintf('%d.',t);
     cCtr = [cCtr; eTotal(t) == sum(eVars(1:t))];
    if t == 1
        P = Polyhedron(lmi(projection(cCtr, eTotal(t))));
        P = Polyhedron('A', full(P.A), 'b', full(P.b), 'Ae', full(P.Ae), 'be', full(P.be));         
        P = Polyhedron('A', [ 1 0; -1 0; 0 1; 0 -1], 'b', [0; 0; max([P.V(1),P.V(2)]); -min([P.V(1),P.V(2)])]);
    else
        P = Polyhedron(lmi(projection(cCtr, [eTotal(t-1), eVars(t)])));
        P = Polyhedron('A', full(P.A), 'b', full(P.b), 'Ae', full(P.Ae), 'be', full(P.be)); 
    end    
    
    % Add the slice
    slices = [slices; P];    
end
   
disp('Done!');

end