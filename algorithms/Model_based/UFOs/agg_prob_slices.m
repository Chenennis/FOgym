function slicesobj = agg_prob_slices(a,b)

slices1 = a.slices;
slices2 = b.slices;
slicespace1 = a.space;
slicespace2 = b.space;
size1 = size(slices1);
numslices = size1(1);
si1 = size1(2);
size2 = size(slices2);
si2 = size2(2);
slicesagg = [];
gr = 1/(slicespace1(2)-slicespace1(1));
slicespaceagg = [slicespace1(1)+slicespace2(1):1/gr:slicespace1(si1)+slicespace2(si2)];
slicelengthagg = size(slicespaceagg);
zagg = min(abs(slicespaceagg));
z1 = min(abs(slicespace1));
z2 = min(abs(slicespace2));
zero = find(abs(slicespaceagg) == zagg);
zerop1 = find(abs(slicespace1) == z1);
zerop2 = find(abs(slicespace2) == z2);
CountMax = si2-zerop2;
for j = 1:numslices
    s2 = repelem(0,slicelengthagg(2));
    for k = 1:slicelengthagg(2)
        s = 0;
        if k < zero
            intmin1 = min(k,zerop1);
            %intmin2 = max(1,k-zerop2);
            %addmax = min([k,zerop2,zero-k,intmin1])-1;
            m1 = min(abs(slicespace2+slicespace1(intmin1)-slicespaceagg(k)));
            intmin2 = find(abs(slicespace2+slicespace1(intmin1)-slicespaceagg(k)) == m1);
            addmax = min(intmin1-1,zerop2-intmin2);
        elseif k == zero
            s = 1;
        else
            intmin1 = max(zerop1,k-zero+zerop1-CountMax);
            %intmin2 = min(k-zero+zerop2,si2);
            %addmax = min([k-zero,si2-zerop2,slicelengthagg(2)-k,slicelengthagg(2)-intmin1]);
            m2 = min(abs(slicespace2+slicespace1(intmin1)-slicespaceagg(k)));
            intmin2 = find(abs(slicespace2+slicespace1(intmin1)-slicespaceagg(k)) == m2);
            addmax = min(si1-intmin1,intmin2-zerop2-1);
        end
        if k ~= zero
            s = slices1(j,intmin1)*slices2(j,intmin2);
            if addmax ~= 0
            for jj = 1:addmax
                if k < zero
                s = s + slices1(j,intmin1-jj)*(slices2(j,intmin2+jj)-slices2(j,intmin2+jj-1));
                else
                s = s + slices1(j,intmin1+jj)*(slices2(j,intmin2-jj)-slices2(j,intmin2-jj+1));
                end
            end
            end
        end
        s2(k) = s;
    end
    slicesagg = [slicesagg;s2];
end
slicesobj = struct;
slicesobj.slices = slicesagg;
slicesobj.space = slicespaceagg;