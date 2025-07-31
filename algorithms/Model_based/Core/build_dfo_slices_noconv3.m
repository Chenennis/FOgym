% This function converts any FlexModel to DFO
function slices = build_dfo_slices_noconv3(obj, mCtr, eVars, NumSamples)
T = size(eVars, 2);

% Inner optimization problem variables
mVars = getvariables(mCtr);

% Additional variables
sdpvar eMin eMax;

% Display message
fprintf('Building DFO slices.');

% Solving options
ops = sdpsettings('verbose',0);%, 'solver','scip'); %, 'bilevel.algorithm','external');
dmin = 0;
dmax = 0;
slices = [];
for t = 1:T
    fprintf('%d.',t);
    
    % Initialize A and b matrices    
    A = [ 1 0; -1 0];
    b = [dmax; -dmin];
    
    if t==1 || dmax - dmin <= 1e-6
        % Solve simple optimization problem
        sol1 = optimize(mCtr, eVars(1),ops);
        vMin = value(eVars(1));
        sol2 = optimize(mCtr, -eVars(1),ops);
        vMax = value(eVars(1));
        
        if sol1.problem ~= 0 || sol2.problem ~= 0
             warning('DFOSystem: infeasible solution during model generation!');
        end
        
        A = [A; 0 1; 0 -1];
        b = [b; vMax; -vMin];
        P = Polyhedron('A', A, 'b', b).normalize().minHRep().minVRep();
    else
        slice = [];
        slice1 = [];
        P = polyshape();
        % Solve bilevel optimization problems
        for s = 0:(NumSamples-1)  
            d = dmin + s * (dmax-dmin) / (NumSamples-1);
                    
            IC = [mCtr, sum(eVars(1:t-1)) == d];
            OO = eMax; % Minimize maximums
            OC = [eMax == eVars(t)];
            IO = [-eVars(t)];
            innerVars = recover(setdiff(mVars, getvariables(eVars(1:t-1))));
            
            solvebilevel(OC,OO,IC,IO,innerVars,ops);            
            vMax = value(eMax);
            
            OO = -eMin; % Maximize minimums
            OC = [eMin == eVars(t)];
            IO = [eVars(t)];
            solvebilevel(OC,OO,IC,IO,innerVars,ops);    
            vMin = value(eMin);
            
             % Add upper and lower bound constraint
            if (s>=1)
                aMin = (vMin - req(2)) / (d - req(1));
                bMin = req(2) - aMin * req(1);                    
                aMax = (vMax - req(3)) / (d - req(1));
                bMax = req(3) - aMax * req(1); 
                
                P1 = polyshape([vMin1(1) vMax1(1) d d],[vMin1(2) vMax1(2) vMax vMin]);%('V', [vMin1; vMax1; d vMin; d vMax]);
                vMin1 = [d vMin];
                vMax1 = [d vMax];
                P = union(P,P1);
                % Check for convexity
                 
                % Add constraints
                if s < 2
                    A = [A; aMin -1];
                    b = [b; -bMin];     
                end         
            
                if s < 2                      
                    A = [A; -aMax 1];
                    b = [b; bMax];
                end
            end
            vMin1 = [d vMin];
            vMax1 = [d vMax];
            % Remember the last energy requirement
            req = [d vMin vMax ];
            slice = [slice; req];
            slice1 = [slice1; d vMin; d vMax];
            fprintf('.');
        end
        A = P.Vertices;
        s1 = size(A);
        rec = 0;
        P0 = Polyhedron();
        for j = 1:s1(1)
        A = circshift(A,1);
        P = polyshape(A);
        B = sides_from_vertices(A);
        add = [];
        Ve = [];
        for k = 1:s1(1)
            if k == s1(1)
                c1 = B(k,1)*B(1,2) - B(k,2)*B(1,1);
            else
                c1 = B(k,1)*B(k+1,2) - B(k,2)*B(k+1,1);
            end
            if sign(c1) == 1
                Ve = [Ve,k];
            end
        end
        b = choose_cut(A,Ve);
        P2 = Polyhedron('V',P.Vertices);
        EQ = P2.H;
        PQ = Polyhedron('H',[EQ;b]);
        if volume(PQ) >= volume(P0)
            P0 = PQ;
        end
        end
        P = P0;
    end
    
    
                
    % We stop, when vMin > vMax
    if P.isEmptySet()
        warning('Cannot generate a valid DFOSystem model. Slice polyhedron is an empty set! Shortening the time window!');
        break;
    end                
    
    % Add the slice
    slices = [slices; P];
    
    [dmin, dmax] = obj.getEnergyTotalMinMax(P);
end

fprintf('Reverse Pass.');

% Performs a reverse pass to make the slices valid (see DFOSystem.isvalid())
for t = fliplr(2 : length(slices))
    [dmin, dmax] = getEnergyDepMinMax(obj, slices(t));
    [tmin, tmax] = getEnergyTotalMinMax(obj, slices(t-1));    
    
    if dmin>tmin+1e-6 || dmax < tmax-1e-6    % Compensate for numerical errors
        % If this happens, just push total constraint to the previous slice
        A = [slices(t-1).A; 1 1; -1 -1];
        b = [slices(t-1).b; dmax; -dmin];       
        slices(t-1) = Polyhedron('A', A, 'b', b).normalize().minHRep().minVRep();
        fprintf('m.');
    end    
end

% Verify and fix the slices after the reverse pass
for t = 1 : length(slices)
   % Check and fix the close-to-LRboundary vertices
   slices(t) = obj.fixSliceVrepresentation(slices(t));
   
   if slices(t).isEmptySet()
       warning('The reverse pass made the slice polyhedron an empty set. Shortening the time window!');
       slices = slices(1:t-1); % Shortening the window
       break;
   end   
end

disp('Done!');

end

