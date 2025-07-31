function slices = build_dfo_slices_outer_artificial(obj, mCtr, eVars, NumSamples)
 T = size(eVars, 2);
 
 % Add additional constraints
% Display message
fprintf('Building DFO slices.');

slices = [];
P1 = Polyhedron([0    5.0000; 0   0]);
slices = [slices,P1];
P1 = Polyhedron([5         -4.5;            5            5;0            5;0 0]);
slices = [slices,P1];
P1 = Polyhedron([       5.2705      -4.7434;           10      -4.7434;           10       4.7816;       9.7816            5;   0            5;   0            0]);
slices = [slices,P1];
P1 = Polyhedron([       5.2705      -4.7434;       14.782      -4.7434;       14.782          0.5;       10.282            5;  0            5;  0   1.9096e-14]);
slices = [slices,P1];
for t = 1:T-4
    P1 = Polyhedron([       5.2705      -4.7434;       14.782+0.5*t      -4.7434;       14.782+0.5*t          0.5;       10.282+0.5*t            5;  -3.5527e-15            5;  -4.4409e-15   1.9096e-14]);
    % Add the slice
    slices = [slices,P1];    
end
   
disp('Done!');

end