function slices = fast_aggregation(DFOs)
sample = DFOs(1).slices;
s = size(sample);
slices = [];
d = size(DFOs);
emin = 0;
emax = 0;
for k = 1:d(2)
    a = DFOs(k).slices;
    v = a(1).V;
    emin = emin + v(2,2);
    emax = emax + v(1,2);
end
P1 = Polyhedron([0    emax; 0   emin]);
slices = [slices, P1];
for t = 2:s(1)
    Vset = repelem(0,6,2);
    count = repelem(0,6);
    for k = 1:d(2)
        a = DFOs(k).slices;
        v = a(t).V;
        m1 = min(v(:,1));
        M1 = max(v(:,1));
        m2 = min(v(:,2));
        M2 = max(v(:,2));
        sv = size(v);        
        for ve = 1:sv(1)
            if v(ve,2) - m2 < 0.0001
                if M1 - v(ve,1) < 0.0001
                    Vset(2,:) = Vset(2,:) + v(ve,:);
                    count(2) = count(2)+1;
                else
                    Vset(1,:) = Vset(1,:) + v(ve,:);
                    count(1) = count(1) + 1;
                end
            elseif M2 - v(ve,2) < 0.0001
                if v(ve,1) - m1 < 0.0001
                    Vset(5,:) = Vset(5,:) + v(ve,:);
                    count(5) = count(5) + 1;
                else
                    Vset(3,:) = Vset(3,:) + v(ve,:);
                    count(3) = count(3) + 1;
                end
            else
                if v(ve,1) - m1 < 0.0001
                    Vset(6,:) = Vset(6,:) + v(ve,:);
                    count(6) = count(6) + 1;
                elseif M1 - v(ve,1)< 0.0001
                    Vset(4,:) = Vset(4,:) + v(ve,:);
                    count(4) = count(4) + 1;
                end
            end
        end
    end
    countok = find(count == d(2));
    P1 = Polyhedron('V',Vset(countok,:));
    slices = [slices,P1];
end