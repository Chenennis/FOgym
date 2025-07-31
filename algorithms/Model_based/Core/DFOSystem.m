classdef DFOSystem < AbstractSystem & FlexSystem
    % This represents a system in dynamic flexoffer representation
    properties(SetAccess=protected)
        % These are 2D polytopes, limiting the input signal with respect to
        % sum of energy, consumed/produced earlier.
        % Each slice must have V (exact) and H (with numerical errors) 
        % representations
        slices             
    end
    
    methods
        
        % Initializes the system using provided slice polyhedrons
        function obj = DFOSystem(varargin)          
            
            % Intialize the FlexSystem
            obj@AbstractSystem();
            obj@FlexSystem();            
            
            if nargin == 0
                error('No model/slices provided as input');
            end
                        
            if nargin == 1 && isa(varargin{1}, 'Polyhedron')     
                
                slices = varargin{1};
            
                % Number of time steps
                N = length(slices);            

                for i=1:N
                    if slices(i).Dim ~= 2
                        error('Each DFO slice must be a 2D polyhedron');
                    end

                    slices(i) = slices(i).minVRep();
                    slices(i) = slices(i).minHRep();
                end
                
                obj.slices = slices;

            elseif nargin >= 1 && isa(varargin{1}, 'struct')
                if nargin <= 3
                    str = varargin{1};
                    eVars = varargin{2};
                    obj.slices = build_dfo_slices_from_struct(str,eVars);
                elseif nargin >= 4 && varargin{4} == "prob"
                    databatt = varargin{1};
                    T = varargin{2};
                    pt = varargin{3};
                    obj.slices = create_prob_DFO_slices(databatt,T,pt);
                end
                
            elseif nargin >= 1 && isa(varargin{1}, 'DFOSystem')
                DFOs = varargin{1};
                obj.slices = fast_aggregation_improved(DFOs);
                
            elseif nargin >= 1 && isa(varargin{1}, 'FlexSystem')
                fm = varargin{1};
                NumSamples = 5;                
                
                if (nargin > 1) 
                    NumSamples = varargin{2};
                end
              
                if ~fm.model.Internal.system.instantiated
                    error('Model is not initialized');
                end
                
                % obj.slices = obj.fmodel_to_dfoslices(fm, NumSamples);                
                if NumSamples == 1
                    % Build a flexoffer 
                    obj.slices = build_dfo_slices_inner_fo_hypersphere(obj, fm.getEnergyPolyhedron);                    
                elseif NumSamples == -1
                    obj.slices = build_dfo_slices_inner_fo_approx(obj, fm.getEnergyPolyhedron());
                elseif NumSamples <= 0
                    eVars = fm.getEnergyVars();                
                    mCtr   = fm.getAllConstraints();
                    obj.slices = build_dfo_slices_outer_approx(obj, mCtr, eVars, NumSamples);  
                elseif NumSamples == 0.5
                    eVars = fm.getEnergyVars();                
                    mCtr   = fm.getAllConstraints();
                    obj.slices = build_dfo_slices_outer_approx_regular(obj, mCtr, eVars, NumSamples);
                elseif NumSamples == 0.1
                    eVars = fm.getEnergyVars();                
                    mCtr   = fm.getAllConstraints();
                    obj.slices = build_dfo_slices_outer_artificial(obj, mCtr, eVars, NumSamples);
                elseif NumSamples == 0.2
                    eVars = fm.getEnergyVars();                
                    mCtr   = fm.getAllConstraints();
                    obj.slices = build_dfo_slices_outer_artificial2(obj, mCtr, eVars, NumSamples);
                elseif NumSamples == 0.3
                    begin = varargin{3};
                    eVars = fm.getEnergyVars();
                    mCtr   = fm.getAllConstraints();
                    obj.slices = build_dfo_slices_outer_analytic(obj, mCtr, eVars, NumSamples,begin);
                elseif NumSamples == 0.301
                    data = varargin{3};
                    eVars = fm.getEnergyVars();
                    mCtr   = fm.getAllConstraints();
                    obj.slices = build_dfo_slices_TEC_onlyup(obj, mCtr, eVars, NumSamples,data);
                elseif NumSamples == 0.31
                    data = varargin{3};
                    obj.slices = build_dfo_slices_outer_analytic_partialbid(data);
                elseif NumSamples == 0.311
                    data = varargin{3};
                    obj.slices = build_dfo_slices_outer_analytic_chargeonly(data);
                elseif NumSamples == 0.312
                    data = varargin{3};
                    obj.slices = build_dfo_slices_heat_constant(data);
                elseif NumSamples == 0.4
                    begin = varargin{3};
                    eVars = fm.getEnergyVars();
                    mCtr   = fm.getAllConstraints();
                    obj.slices = build_dfo_slices_outer_analytic_uncertainoldpaper(obj, mCtr, eVars, NumSamples,begin);
                elseif NumSamples == 0.9
                    data = varargin{3};
                    obj.slices = build_dfo_slices_heat_MES(data);                
                elseif NumSamples == 1.1
                    FO = varargin{3};
                    COP = varargin{4};
                    obj.slices = convert_HFO_to_DFO(FO,COP);
                elseif nargin >= 3 && varargin{3} == "noconv"
                    eVars = fm.getEnergyVars();
                    mCtr   = fm.getAllConstraints();
                    obj.slices = build_dfo_slices_noconv(obj, mCtr, eVars, NumSamples);
                elseif nargin >= 3 && varargin{3} == "noconv2"
                    eVars = fm.getEnergyVars();
                    mCtr   = fm.getAllConstraints();
                    obj.slices = build_dfo_slices_noconv2(obj, mCtr, eVars, NumSamples);
                elseif nargin >= 3 && varargin{3} == "noconv3"
                    eVars = fm.getEnergyVars();
                    mCtr   = fm.getAllConstraints();
                    obj.slices = build_dfo_slices_noconv3(obj, mCtr, eVars, NumSamples);
                else
                    % Normally build a DFO system 
                    %obj.slices = build_dfo_slices_optim_rev(obj, fm.getEnergyPolyhedron, NumSamples);                    
                    eVars = fm.getEnergyVars();                
                    mCtr   = fm.getAllConstraints();
                    %obj.slices = build_dfo_slices_approx(obj, mCtr, eVars, NumSamples);   
                    obj.slices = build_dfo_slices(obj, mCtr, eVars, NumSamples);   
                end
            else
                error('Unsupported input!');            
            end     
                      
            obj.nx = 1;
            obj.nu = 1;
            obj.ny = 1;
            
            % State signal is "total energy"
            x = SystemSignal(obj.nu);
            x.name = 'Total energy';
            x.setKind('x');
            obj.addComponent('x', x);    
            
            % Input signal is "slice energy"
            u = SystemSignal(obj.nu);
            u.name = 'Input Energy';
            u.setKind('u');
            obj.addComponent('u', u);
            
            % Output signal is "slice energy"
            y = SystemSignal(obj.ny);
            y.name = 'Output Energy';
            y.setKind('y');
            obj.addComponent('y', y);
            
            % Intantiate the system
            obj.initialize(0);
            obj.instantiate(length(obj.slices));
            
            % Initialize the FlexSystem
            obj.init_fs(obj, 1:length(obj.slices));
        end
        
        function C = constraints(obj)
            % Convert LTI model into YALMIP constraints            
            C = constraints@AbstractSystem(obj);
            
            % add the LTI dynamics constraints
            x = obj.x.var;
            u = obj.u.var;
            y = obj.y.var;
            
            % Initial conditions
            C = C + [ x(1, 1) == 0 ];
         
            for k = 1:obj.Internal.system.N
                % Add polytope constraints
                % A = obj.slices(k).A;
                % b = obj.slices(k).b;   
                v = [x(1, k); u(1, k)];
                C = C + [ ismember(v, obj.slices(k)) ]; 
                % A * [x(1, k); u(1, k)] <= b
                
                % Add tota energy transfer constraint
                C = C + [ x(1, k+1) == x(1, k) + u(1, k) ];                
                % Add output signal constraint                
                C = C + [ y(1, k) == u(1, k) ];                 
            end
        end
        
        % Overloaded more efficient method for retrieving a DFO polyhedron
       %  function P = getEnergyPolyhedron(obj)
       %     A = 
       %     P = Polyhedron('A', full(Ppold.A), 'b', full(Ppold.b));            
       % end
       
        % Compute DFOSystem outer approximation
        function dfo_outer = outerApprox(obj)
            dfo_outer = DFOSystem(obj.outerSliceApprox(obj.slices));
        end
        
        % Compute absolute flexibility approximatelly
        function f = computeAbsoluteFlexibilityApprox(obj)           
           [emin, emax] = obj.getEnergyMinMax(obj.slices(1), 0);
           f = emax - emin;
           for t = 2 : length(obj.slices)
                % [dlmin, dlmax] = obj.getEnergyDepMinMax(obj.slices(t-1));
                [dmin, dmax] = obj.getEnergyDepMinMax(obj.slices(t));
                m =  obj.slices(t).volume() / (dmax - dmin);%  * (dmax - dmin - dlmax + dlmin);
                f = f * m;
           end            
        end        
        
        % Compute average dimension flexibility approximatelly
        function f = computeAvgDimensionFlexibilityApprox(obj)           
           [emin, emax] = obj.getEnergyMinMax(obj.slices(1), 0);
           f = emax - emin;
           for t = 2 : length(obj.slices)
                % [dlmin, dlmax] = obj.getEnergyDepMinMax(obj.slices(t-1));
                [dmin, dmax] = obj.getEnergyDepMinMax(obj.slices(t));
                m =  obj.slices(t).volume() / (dmax - dmin);%  * (dmax - dmin - dlmax + dlmin);
                f = f + m;
           end         
           f = f / length(obj.slices);
        end                

        
        % Compute flexibility, ralative to another reference model
        % If reference model is not specified, using obj's outer
        % approximation
        function f = computeRelativeFlexibilityApprox(obj, model_ref)
            
           if nargin < 2
               model_ref = obj.outerApprox();
           end
            
           if ~isa(model_ref, 'DFOSystem')
               error('To compute flexibility, the reference system must be of DFOSystem type');
           end
           
           if length(obj.slices) ~= length(model_ref.slices)
               error('To compute flexibility, the number of slices must be equal');
           end
           
           f = obj.computeSliceFlexibility(model_ref.slices);
        end
        
        % This method performs disaggregation using optimization
        function d_err = disaggregate(obj, subFSs)
            if nargin >= 2
                % Use (default) optimization-based disaggregation
                d_err = obj.disaggregate@FlexSystem(subFSs);
            else
                % Else disaggregate to itself, as DFOSystem is not a composite model
                warning('DFOSystem disaggregates to itself, as it is not a composite system!');
                d_err = 0;                
            end
        end
        
        % This method checks if DFOSystem slices are OK
        function b = is_valid(obj)
            b = false;
            
            % Check for time horizon coverage
            if length(obj.slices) ~= obj.getNumTimeSteps()
               warning('DFOSystem initialized time and slice count do not match!');
               return;
            end
            
            % Check each slice
            dmin = 0;
            dmax = 0;
            for t = 1 : length(obj.slices) 
                
                % Check for emptiness
                if obj.slices(t).isEmptySet
                    warning('DFOSystem slice is empty set!');
                    return;
                end
                
                [sdmin, sdmax] = obj.getEnergyDepMinMax(obj.slices(t));
                [stmin, stmax] = obj.getEnergyTotalMinMax(obj.slices(t));
                
                % Check 1st slice
                if t == 1 && (sdmin ~= 0 || sdmax ~= 0)
                    warning('DFOSystem first slice has invalid dependent energy. Must be 0');
                    return;
                end
                
                % Check the dependent energy coverage
                if sdmin-1e-5 > dmin || sdmax + 1e-5 < dmax
                    warning('DFOSystem: The dependent energy range is not covered!');
                    return;
                end
                
                % Check total min/max - hardly possible
                if stmin > stmax
                    warning('DFOSystem total maximum energy is lower then total minimum energy!');
                    return;
                end
                
                % Check the H representation: A matrix
                A = obj.slices(t).A;
                Ae = obj.slices(t).Ae;
                
                if all(abs(A(:,2)) <= 1e-6) && (isempty(Ae) || all(abs(Ae(:, 2)) <= 1e-6))
                    warning('DFOSystem: A matrix allows energy to be unbounded! ');
                    return;                                        
                end
                
                if abs(sum(A.^2,2)-1)>1e-6
                    warning('DFOSystem: A matrix of a slice is not normalized! Call ".normalize()".');
                    return;                    
                end       
                
                %if any(abs(A(:, 2))<=1e-6 & A(:, 2) ~= 0)
                %    warning('DFOSystem: A matrix of slice has close-to-zero energy components!');
                %    return;
                %end
                                            
                % Check the V representation: min/max amounts            
                Vd = obj.slices(t).V(:,1);
                if any(abs(Vd - sdmin) <= 1e-6 & Vd ~= sdmin) ||...
                  any(abs(Vd - sdmax) <= 1e-6 & Vd ~= sdmax)
                  warning('DFOSystem: V matrix of slice has close-to-LRboundary vertices!');
                  return;                    
                end                
                
                D = obj.slices(t).V(:, 1);
                for d = 1 : length(D)
                     [semin, semax] = obj.getEnergyMinMax(obj.slices(t), D(d));
                     if (semin > semax)
                         warning('DFOSystem: Minimum energy is higher than maximum energy! Perhaps dependent energy range is invalid!');
                         return;
                     end
                end
                
                dmin = stmin;
                dmax = stmax;
            end
            b = true;
        end
        
        
        % This computes inner approximation (FlexOffer) of a DFOSystem
        function dfoInner = computeInnerApproximation(obj)
            ops = sdpsettings('verbose',0);
            
            T = length(obj.slices);
            eMin = sdpvar(1, T);
            eMax = sdpvar(1, T);
            
            C = [];
            O = 0;
            for t = 1 : T
             C = [C, eMin(t) <= eMax(t)];             
             C = [C; ismember( [sum(eMin(1:t-1)); eMin(t)], obj.slices(t));...
                     ismember( [sum(eMax(1:t-1)); eMax(t)], obj.slices(t))];
            
             O = O + (eMax(t) - eMin(t));
            end            
            
            sol = optimize(C, -O, ops);
            
            if sol.problem ~= 0
                error('Failed finding the inner approximation!');
            end
            
            % Build the inner approximation DFOSystem
            slicesI = [];
            dmin = 0;
            dmax = 0;
            for t = 1 : T 
                slicesI = [slicesI; Polyhedron('V', [dmin value(eMin(t));
                                                     dmax value(eMin(t));
                                                     dmax value(eMax(t));
                                                     dmin value(eMax(t))]).minHRep().normalize()];
                dmin = dmin + value(eMin(t));
                dmax = dmax + value(eMax(t));
            end
            dfoInner = DFOSystem(slicesI);
        end
    end   
    
    methods(Hidden, Access = protected)
        % DFO aggregation technique
        function state0 = aggregateInit(obj, params)     
            if ~isstruct(params)
                error('Aggregation parameters must be of struct type!');
            end
            
            if isfield(params, 'deterministic')
                disp('Using deterministic DFOSystem aggregation');
                % Perform deterministic aggregation
                state0 = aggregate_determ_init(obj, params);
            else
                disp('Using exact DFOSystem aggregation');
                state0 = obj;
            end
        end
        
        function state = aggregateTrans(obj, state, next_fs, params)
            
            if isfield(params, 'deterministic')               
                % Perform deterministic aggregation
                state = aggregate_determ_trans(obj, state, next_fs, params);
            else
	            % Exact aggregation
				sample_count = 5; 
				if isfield(params, 'sample_count') 
					sample_count = params.sample_count;
				end     
                state = DFOSystem(MultiSystem([state; next_fs]), sample_count);
            end
        end
        
        function agg_model = aggregateFinal(obj, state, params)                        
            if isfield(params, 'deterministic')                    
               % Perform deterministic aggregation
               agg_model = aggregate_determ_final(obj, state, params);
            else
               if isempty(state)
                    error('Cannot aggregate empty set of FlexSystems');
               end
               agg_model = state;
            end            
        end
        
        function out = display_internal(obj)
            % Returns a string information about the system            
            plural = @(s, n) [num2str(n) ' ' s repmat('s', 1, double(n~=1))];
            out = sprintf('%s with %s, %s, %s', class(obj), ...
					plural('state', obj.nx), ...
					plural('input', obj.nu), ...
					plural('output', obj.ny));
        end
    end
     
     methods(Hidden, Access = protected)
        
         
         function [xn, y, varargout] = state_equation(obj, x, u, varargin)

		% Output equation.
		%
		% Mandatory inputs:
		%   obj: the system object
		%     x: the state vector
		%     u: the input vector
        % Optional inputs:
        %   parameters
        % Mandatory outputs:
		%     y: system's output
		y = output_equation(obj, x, u, varargin)
     end
         % Implementation of abstract methods
        function [xn, y] = update_equation(obj, x, u)
            xn = x + u;     % Sum total energy
            y = u;
        end
        
        function y = output_equation(obj, x, u)
            y = u; % In DFO, response function is identity. No errors are considered.
        end
        
        function out = has_feedthrough(obj)
            % feedthrough indication. must return true if the system has
            % direct feedthrough, false otherwise
            out = true;
        end     
         
         % ******* Private methods **************
         % Deterministically get's range of energy, for a given slice and dependent energy
         function [e_min, e_max] = getEnergyMinMax(obj, slice, en_dep)
             V = slice.V;
             A = slice.A;
             b = slice.b;
             Ae = slice.Ae;
             be = slice.be;
             
             dep_min = min(V(:, 1)); 
             dep_max = max(V(:, 1)); 
             
             % Check the boundary cases
             if (en_dep <= dep_min)
                 e_min = min(V(find(abs(V(:,1) - dep_min) <= 1e-6),2));
                 e_max = max(V(find(abs(V(:,1) - dep_min) <= 1e-6),2));
             elseif (en_dep >= dep_max)
                 e_min = min(V(find(abs(V(:,1) - dep_max) <= 1e-6),2));
                 e_max = max(V(find(abs(V(:,1) - dep_max) <= 1e-6),2));
             else                 
                 % All normal cases
                 e_min = -inf;
                 e_max = inf;

                 for i = 1 : size(A, 1)                     
                     if A(i, 2) > 1e-6
                        e_max = min(e_max, (b(i) - A(i, 1) * en_dep) / A(i, 2));
                     elseif A(i, 2) < -1e-6
                        e_min = max(e_min, (b(i) - A(i, 1) * en_dep) / A(i, 2));
                     end                    
                 end
                 
                 % Equality constraints are set, so check them also
                 if ~isempty(Ae)
                     for i = 1 : size(Ae, 1)                          
                         e_max = min(e_max, (be(i) - Ae(i, 1) * en_dep) / Ae(i, 2));
                         e_min = max(e_min, (be(i) - Ae(i, 1) * en_dep) / Ae(i, 2));
                     end
                 end                 
             end
         end
        
         % Get minimum/maximum dependend energy of the slice
         function [e_min, e_max] = getEnergyDepMinMax(obj, slice)
             V = slice.V;
             e_min = min(V(:,1));
             e_max = max(V(:,1));
         end
         
         % Get minimum/maximum total energy of the slice
         function [e_min, e_max] = getEnergyTotalMinMax(obj, slice)
             V = slice.V;
             e_min = min(V(:,1) + V(:,2));
             e_max = max(V(:,1) + V(:,2));
         end         
         
         % Fix the V representation of a slice, to avoid the LR
         % boundary problem
         function slice = fixSliceVrepresentation(obj, slice)
               [dmin, dmax] = getEnergyDepMinMax(obj, slice);
               V = slice.V;   
               ErrsMin = abs(V(:,1) - dmin) <= 1e-6 & V(:,1) ~= dmin;
               ErrsMax = abs(V(:,1) - dmax) <= 1e-6 & V(:,1) ~= dmax;
               if any(ErrsMin) || any(ErrsMax)       
                   V(find(ErrsMin), 1) = dmin;
                   V(find(ErrsMax), 1) = dmax;
                   slice = Polyhedron('V', V).minVRep().minHRep().normalize();
%                   fprintf('f.');
               end
         end
         
        % Gets outter approximation of DFOSystem slices
        function approxSlices = outerSliceApprox(obj, slices)
            approxSlices = arrayfun(@(slice) slice.outerApprox(), slices, 'UniformOutput', false);
            approxSlices = [approxSlices{:}];
        end
    
         % Compute flexibility, ralative to reference slices
        function f = computeSliceFlexibility(obj, slices_ref)           
            f = 1;
            for t = 1 : length(obj.slices)
                if t == 1
                    [emin, emax] = obj.getEnergyMinMax(obj.slices(t), 0);
                    [rmin, rmax] = obj.getEnergyMinMax(slices_ref(t), 0);
                    if abs(rmax - rmin)>=1e-5
                        f = f * (emax - emin) / (rmax - rmin);
                    end
                else
                    ref_vol = slices_ref(t).volume();
                    if abs(ref_vol) >= 1e-5
                        f = f * (obj.slices(t).volume() / ref_vol);
                    end
                end
            end
        end
        
        % Compute flexibility with respect to its outer approximation
        function f = computeSliceOuterApproxFlexibility(obj, slices)
            approxSlices = obj.outerSliceApprox(slices);
            f = obj.computeSliceFlexibility(approxSlices);
        end
         
     end
end

%          
%          % Get minimum/maximum energy of a given slice
%          function [e_min, e_max] = getMinMaxTotalEnergy(obj, slice)
%              
%          end
%          
%          % Get all distinct dependent energy combinations of a polytope
%          function r = distinctDepEnergyComponents(obj, slice)
%              r = sort(unique(slice.V(:,1)));
%          end   

%if ~P.hasVRep
    % error('Each DFO slice polyhedron must have a V-representation');                
%end
%if ~P.hasHRep
 %   slices(i) = P.minHRep();
    % error('Each DFO slice polyhedron must have a H-representation');
%end


%              dep_min = min(slice.V(:, 1)); 
%              dep_max = max(slice.V(:, 1)); 
%              
%              if (en_dep <= dep_min)
%                  e_min = min(slice.V(find(slice.V(:,1) == dep_min),2));
%                  e_max = max(slice.V(find(slice.V(:,1) == dep_min),2));
%              elseif (en_dep >= dep_max)
%                  e_min = min(slice.V(find(slice.V(:,1) == dep_max),2));
%                  e_max = max(slice.V(find(slice.V(:,1) == dep_max),2));
%              else                 
%                  e_min = NaN;
%                  e_max = NaN;
%                  
%                  for i = 1 : length(slice.V)
%                      ii = mod(i, length(slice.V))+1;
%                      if slice.V(i,1) <= en_dep && en_dep <= slice.V(ii,1)
%                          v = slice.V(i,2) + (slice.V(ii,2) - slice.V(i,2)) * (en_dep - slice.V(i,1)) / (slice.V(ii,1) - slice.V(i,1));
%                          e_min = min(e_min, v);
%                          e_max = max(e_max, v);
%                      end
%                      
%                      if slice.V(ii,1) <= en_dep && en_dep < slice.V(i,1)
%                          v = slice.V(ii,2) + (slice.V(i,2) - slice.V(ii,2)) * (slice.V(ii,1) - en_dep) / (slice.V(ii,1) - slice.V(i,1));
%                          e_min = min(e_min, v);
%                          e_max = max(e_max, v);
%                      end
%                  end
%              end
%                 % Check if energy bounds can be computed equally using
%                 % the deterministic technique and MPT's library
%                 for d = dmin : (dmax-dmin)/15 : dmax
%                       [semin, semax] = obj.getEnergyMinMax(obj.slices(t), d);
%                       S = obj.slices(t).slice(1, d);
%                       rmin = min(S.V);
%                       rmax = max(S.V);
%                       
%                       if abs(semin-rmin)>1e-6 || abs(semax-rmax)>1e-6
%                          warning('DFOSystem: MIN/MAX bounds cannot be computed correctly! ');
%                          return;                          
%                       end                      
%                 end
