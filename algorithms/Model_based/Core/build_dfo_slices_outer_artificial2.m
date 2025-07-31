function slices = build_dfo_slices_outer_artificial2(obj, mCtr, eVars, NumSamples)
 T = size(eVars, 2);
 
 % Add additional constraints
% Display message
fprintf('Building DFO slices.');

slices = [];
P1 = Polyhedron([0          6.7;            0            0]);
slices = [slices,P1];
P1 = Polyhedron([6.7      -6.4655;          6.7          6.7;0          6.7;  0 0]);
slices = [slices,P1];
P1 = Polyhedron([       6.8204      -6.5817;         13.4      -6.5817;         13.4          6.7;0          6.7;0 0]);
slices = [slices,P1];
P1 = Polyhedron([6.8204      -6.5817;         20.1      -6.5817;         20.1      0.48488;       13.885          6.7;  0          6.7;  0   0]);
slices = [slices,P1];
for t = 1:T-4
    P1 = Polyhedron([6.8204      -6.5817;         20.1+0.48488*t      -6.5817;         20.1+0.48488*t      0.48488;       13.885+0.48488*t          6.7;  0          6.7;  0   0]);
    % Add the slice
    slices = [slices,P1];    
end
   
disp('Done!');

end