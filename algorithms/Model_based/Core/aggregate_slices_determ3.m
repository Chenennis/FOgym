function [agg_prec1, agg_prec2, slicesA] = aggregate_slices_determ3( obj, slices1, slices2, NumSamples )
% This function deterministically aggregate DFOSystem slices
%   "agg_prec1/agg_prec_2" represents the precision 0(low)..1(high), at
%   which slices1/slices2 are included into the aggregate

N = max(length(slices1), length(slices2));

if length(slices1) ~= length(slices2)
    error('Aggregating DFOSystems with different number of slices!');      
end

slicesA = [];
dminA = 0;
dmaxA = 0;
agg_prec1 = 1;
agg_prec2 = 1;

for t = 1:N
 slice1 = slices1(t);
 slice2 = slices2(t);
 sliceA = [];
 sliceAconvex = true;
 
 % Compute dependent energy minimum/maximum
 [dmin1, dmax1] = obj.getEnergyDepMinMax(slice1);
 [dmin2, dmax2] = obj.getEnergyDepMinMax(slice2);

 % Compute total minimum/maximum
 [tmin1, tmax1] = obj.getEnergyTotalMinMax(slice1);
 [tmin2, tmax2] = obj.getEnergyTotalMinMax(slice2);
 
 if nargin >= 4 && NumSamples > 0
    % Manually set ratios
    R = 0: 1/NumSamples : 1;
 else
    % Automatically compute unique rations entailed by slice1 and slice2 dep. energies
    NumSamples = 0;    
    R = [0; 1]; 
    if dmax1>dmin1
         R = [R; (slice1.V(:,1)-dmin1)/(dmax1 - dmin1) ];
    end 
    if (dmax2>dmin2)
         R = [R; (slice2.V(:,1)-dmin2)/(dmax2 - dmin2) ];
    end 
    R = sort(unique(R)); % Filter unique rows 
 end

agg_prec1a = 0;
agg_prec2a = 0;
agg_prec1e = 0;
agg_prec2e = 0;

 
 for i = 1:length(R)
     dA = dminA + R(i)*(dmaxA - dminA);
     d1 = dmin1 + R(i)*(dmax1 - dmin1);
     d2 = dmin2 + R(i)*(dmax2 - dmin2);
     
     [eMin1 eMax1] = obj.getEnergyMinMax(slice1, d1);
     [eMin2 eMax2] = obj.getEnergyMinMax(slice2, d2);
     
     % Solve the following 2 optimization problems deterministically
     % C = [d1 + eMin1 <= t1 <= d1 + eMax1];
     % C = [d2 + eMin2 <= t2 <= d2 + eMax2; C];
     % C = [(t1-tmin1)/(tmax1 - tmin1) == (t2 - tmin2) / (tmax2 - tmin2); C];
     % sol1 = optimize(C,   t1+t2, ops);     
     % sol2 = optimize(C,  -t1-t2, ops);     
     
     % Here:
     %  t1 = tmin1 + (tmax1 - tmin1) * (t2 - tmin2) / (tmax2 - tmin2)
     %  t2 = tmin2 + (tmax2 - tmin2) * (t1 - tmin1) / (tmax1 - tmin1)
     
     if abs(tmax1 - tmin1)<= 1e-5 || abs(tmax2 - tmin2)<= 1e-5 || t == N || NumSamples == 1 % FlexOffer
         eMinA = eMin1 + eMin2;
         eMaxA = eMax1 + eMax2;   
         agg_prec1a = agg_prec1a + (eMax1 - eMin1);
         agg_prec2a = agg_prec2a + (eMax2 - eMin2);
     else

         % Vector of potential solutions
         t1_1 = d1 + eMin1;
         t2_1 = min(d2 + eMax2, tmin2 + (tmax2 - tmin2) * (d1 + eMin1 - tmin1) / (tmax1 - tmin1));

         t2_2 = d2 + eMin2;
         t1_2 = min(d1 + eMax1, tmin1 + (tmax1 - tmin1) * (d2 + eMin2 - tmin2) / (tmax2 - tmin2));

         if (t1_1+t2_1 > t1_2 + t2_2)
            ttMin = t1_1 + t2_1;

            if abs((t1_1-tmin1)/(tmax1 - tmin1) - (t2_1 - tmin2) / (tmax2 - tmin2)) > 1e-4
                % warning('Aggregation problem is infeasible!');
                slicesA = []; agg_prec1 = 0; agg_prec2 = 0;
                return;
            else
                agg_prec1a = agg_prec1a - t1_1;
                agg_prec2a = agg_prec2a - t2_1;
            end
         else
            ttMin = t1_2 + t2_2;

            if abs((t1_2-tmin1)/(tmax1 - tmin1) - (t2_2 - tmin2) / (tmax2 - tmin2)) > 1e-4
                % warning('Aggregation problem is infeasible!');
                slicesA = []; agg_prec1 = 0; agg_prec2 = 0;
                return;
            else
                agg_prec1a = agg_prec1a - t1_2;
                agg_prec2a = agg_prec2a - t2_2;
            end
         end

         t1_1 = d1 + eMax1 ;
         t2_1 = max(d2 + eMin2, tmin2 + (tmax2 - tmin2) * (d1 + eMax1 - tmin1) / (tmax1 - tmin1));

         t2_2 = d2 + eMax2;
         t1_2 = max(d1 + eMin1, tmin1 + (tmax1 - tmin1) * (d2 + eMax2 - tmin2) / (tmax2 - tmin2));

         if (t1_1+t2_1 < t1_2 + t2_2)
            ttMax = t1_1 + t2_1;

            if abs((t1_1-tmin1)/(tmax1 - tmin1) - (t2_1 - tmin2) / (tmax2 - tmin2)) > 1e-4
                % warning('Aggregation problem is infeasible!');
                slicesA = []; agg_prec1 = 0; agg_prec2 = 0;
                return;
            else
                agg_prec1a = agg_prec1a + t1_1;
                agg_prec2a = agg_prec2a + t2_1;                
            end
         else
            ttMax = t1_2 + t2_2;

            if abs((t1_2-tmin1)/(tmax1 - tmin1) - (t2_2 - tmin2) / (tmax2 - tmin2)) > 1e-4
                % warning('Aggregation problem is infeasible!');
                slicesA = []; agg_prec1 = 0; agg_prec2 = 0;
                return;
            else
                agg_prec1a = agg_prec1a + t1_2;
                agg_prec2a = agg_prec2a + t2_2;                   
            end
         end         
         
         eMinA = ttMin - dA;
         eMaxA = ttMax - dA;
     end
     
     % Check for convexity
     if (i >= 3)
         mMin = (sliceA(i-1,2)-sliceA(i-2,2))/ ...
                (sliceA(i-1,1)-sliceA(i-2,1)) * ...
                (dA - sliceA(i-2,1)) + sliceA(i-2,2);
         mMax = (sliceA(i-1,3)-sliceA(i-2,3))/ ...
             (sliceA(i-1,1)-sliceA(i-2,1)) * ...
             (dA - sliceA(i-2,1)) + sliceA(i-2,3);                                
         
         if (eMinA<mMin-1e-4)
             warning('An aggregate results in a non-convex slice!');
             %eMinA = mMin;
             sliceAconvex = false;
         end
         if (eMaxA>mMax+1e-4)        
             warning('An aggregate results in a non-convex slice!');
             %eMaxA = mMax;              % slicesA = []; agg_prec1 = 0; agg_prec2 = 0;
             sliceAconvex = false;
         end                  
     end
      
     if (eMinA>eMaxA)
            warning('Upps. eMin>eMax! Cannot deterministically aggregate DFOSystems. Make sure DFOSystem are valid! ');
            eMaxA = eMinA;
     end
     
     % Update an error measures
     agg_prec1e = agg_prec1e + (eMax1 - eMin1);
     agg_prec2e = agg_prec2e + (eMax2 - eMin2);
     %agg_prec_12 = agg_prec_12 + (eMax1 - eMin1) + (eMax2 - eMin2);
     %agg_prec_A + eMaxA - eMinA;
     % if (d12>0)
     %   agg_prec_run = agg_prec_run + (eMaxA - eMinA) / d12;
     %   agg_prec_cnt = agg_prec_cnt + 1;
     %end
    
     % Generate slice
     sliceA = [sliceA; dA eMinA eMaxA];
 end
 
  if nargout > 2
      % Slice output requested
      if sliceAconvex
          % For performance reasons, make a dummy Polyhedron
          [~, idx] = unique(sliceA(:,1));
          sliceA = sliceA(idx, :);

          A = [ 1 0; -1 0];
          b = [sliceA(end,1); -sliceA(1,1)];

          for i = 2 : size(sliceA, 1)
                aMin = (sliceA(i,2) - sliceA(i-1,2)) / (sliceA(i,1) - sliceA(i-1,1));
                bMin = sliceA(i-1,2) - aMin * sliceA(i-1, 1);                    
                aMax = (sliceA(i,3) - sliceA(i-1,3)) / (sliceA(i,1) - sliceA(i-1,1));
                bMax = sliceA(i-1,3) - aMax * sliceA(i-1, 1); 

                A = [A; aMin -1; -aMax 1];
                b = [b; -bMin; bMax];                 
          end
          P = struct('V', [sliceA(:, 1) sliceA(:, 2); sliceA(:, 1) sliceA(:, 3)],...
                                 'A', A, 'b', b, 'Ae', [], 'be', []);
      else
          Poly = Polyhedron([sliceA(:, 1) sliceA(:, 2); sliceA(:, 1) sliceA(:, 3)]).computeHRep();
          P = struct('V', [Poly.V], 'A', Poly.A, 'b', Poly.b, 'Ae', Poly.Ae, 'be', Poly.be); 
      end
      
      slicesA = [slicesA; P];      
      
      %slicesA = [slicesA; Polyhedron([sliceA(:, 1) sliceA(:, 2);
      %                                sliceA(:, 1) sliceA(:, 3)]).computeHRep()];
  end
  dminA = min(sliceA(:,1)+sliceA(:,2));
  dmaxA = max(sliceA(:,1)+sliceA(:,3));  
  
  % Update an overall error
  if agg_prec1e > 0
       agg_prec1 = agg_prec1 * agg_prec1a / agg_prec1e;
  end
  if agg_prec2e > 0
       agg_prec2 = agg_prec2 * agg_prec2a / agg_prec2e;
  end  
end


end

