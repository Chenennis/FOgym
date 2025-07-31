function slicestructure=create_prob_slices_charging(d,t,gr)

%TO BE REWORKED
avgs = [];
Cmax = d.L*d.Pmax;

Q = [];
totalspace = [d.Qmin:1/gr:d.Qmax];
length = size(totalspace);

%Generating probability distribution for Q

slicezero = repelem(0,length(2));
slicezero(d.Q0*gr+1) = length(2);
Q = [Q,slicezero];

s1 = repelem(0,length(2));
lmin = d.Q0;
lmax = min(Qmax,d.Q0+Cmax);
s1(lmin*gr+1:Q0*gr) = 0;
s1(Q0*gr+2:lmax*gr) = 1/(lmax-d.Q0);
%s1(lmin*gr+1:lmax*gr) = 1/(lmax-lmin);
Q=[Q;s1];
CountMax = floor(Cmax*gr);
CountMin = 0;

for j = 1:t-2
    s2 = repelem(0,length(2));
    total = sum(s1);
    for k = 1:length(2)
        ccmax = max(d.Qmin*gr+1,k-CountMax);
        ccmin = min(d.Qmax*gr+1,k-CountMin);
        s2(k) = sum(s1(ccmax:ccmin))/total;
    end
    ss = sum(s2);
    s2 = s2*gr/(ss);
    Q = [Q;s2];
    s1 = s2;
%    plot(totalspace,s2)
end

si = size(Q);
for k = 1:si(1)
ev = 0;
for j = 1:si(2)
ev = ev + j*Q(k,j)/gr;
end
ev = ev/gr;
end
avgs = [avgs,ev];

%Generating slices

slices = [];
slicespace = [d.Pmin:1/gr:Cmax];
slicelength = size(slicespace);
tt = -d.Pmin*gr + 1;
for j = 1:t
    slice = repelem(0,slicelength(2));
    QS = Q(j,:);
    for k = 1:slicelength(2)
        slice(k) = sum(QS(max(tt-k,1):min(length(2),length(2)+tt-k)))/sum(QS);
    end
    slices=[slices;slice];
end
slicestructure = struct;
slicestructure.slices = slices;
slicestructure.space = slicespace;