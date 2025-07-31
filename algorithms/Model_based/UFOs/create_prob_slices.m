function slicestructure=create_prob_slices(Qmin,Qmax,Q0,Pmin,Pmax,L,numslices,gr)

avgs = [];
Cmax = L*Pmax;

Q = [];
totalspace = [Qmin:1/gr:Qmax];
length = size(totalspace);

%Generating probability distribution for Q

slicezero = repelem(0,length(2));
slicezero(Q0*gr+1) = length(2);
Q = [Q,slicezero];

s1 = repelem(0,length(2));
lmin = max(Qmin,Q0+Pmin);
lmax = min(Qmax,Q0+Cmax);
s1(lmin*gr+1:Q0*gr) = 0.5/(Q0-lmin);
s1(Q0*gr+2:lmax*gr) = 0.5/(lmax-Q0);
%s1(lmin*gr+1:lmax*gr) = 1/(lmax-lmin);
Q=[Q;s1];
CountMax = floor(Cmax*gr);
CountMin = floor(Pmin*gr);

for j = 1:numslices-2
    s2 = repelem(0,length(2));
    total = sum(s1);
    for k = 1:length(2)
        ccmax = max(Qmin*gr+1,k-CountMax);
        ccmin = min(Qmax*gr+1,k-CountMin);
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
slicespace = [Pmin:1/gr:Cmax];
slicelength = size(slicespace);
tt = -Pmin*gr + 1;
for j = 1:numslices
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
