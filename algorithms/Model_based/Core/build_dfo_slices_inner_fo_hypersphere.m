function slices = build_dfo_slices_inner_fo_hypersphere(obj, poly)
            % Build the maximal inner FO approximation
            ops = sdpsettings('verbose',0);
            
            T = poly.Dim;
            % fprintf('Display poly');
            % disp(value(poly.V));
            % Compute the Chebyshev's ball
            C = poly.chebyCenter();
            
            
            % Build the inner approximation DFOSystem
            slices = [];
            dmin = 0;
            dmax = 0;
            for t = 1 : T 
                x_min = C.x(t) - sqrt(C.r^2/T);
                x_max = C.x(t) + sqrt(C.r^2/T);
                
                slices = [slices; Polyhedron('V', [dmin x_min;
                                                     dmax x_min;
                                                     dmax x_max;
                                                     dmin x_max]).minHRep().normalize()];
                dmin = dmin + x_min;
                dmax = dmax + x_max;
            end
end