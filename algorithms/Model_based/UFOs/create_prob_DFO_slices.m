function slices = create_prob_DFO_slices(a,T,pt)
Qmin = a.Qmin;
Qmax = a.Qmax;
Q0 = a.InitialSoC;
Pmin = a.Pmin;
Pmax = a.Pmax;
L = a.L;
numslices = T;

%set first slice

summin = repelem(0,numslices); %sum lower bound
summax = repelem(0,numslices); %sum upper bound
vup = repelem(0,numslices); %upper vertex x coordinate
vdown = repelem(0,numslices); %down vertex x coordinate
emin = repelem(0,numslices); %DFO lower bound
emax = repelem(0,numslices); %DFO upper bound

%building slices
slices = [];

%first slice

emin(1) = max(Qmin-Q0,Pmin);
emax(1) = min(Qmax-Q0,Pmax);
slices = [slices,Polyhedron([0,emin(1);0,emax(1)])];

%other slices

for t = 2:numslices
    summin(t) = summin(t-1)+emin(t-1);
    summax(t) = summax(t-1)+emax(t-1);
    emin(t) = max(-Q0-summin(t),Pmin);
    %calculating emax
    if Q0 > Qmax - L*Pmax
        emax(t) = 0;
    else
        for a = 1:t-1
            if Q0 + a*Pmax + (t-1-a)*Pmin >= summax(t)
                break
            end
        end
        QQmin = max(Q0 + a*(L-1)*Pmax + summax(t),0);
        QQmax = Q0 + summax(t)*L;
        emax(t) = min((Qmax - pt* QQmax + (pt-1)*QQmin)/L,Pmax);
    end
    vup(t) = summax(t) + L*(emax(t) - Pmax);
    vdown(t) = summin(t) + emin(t) - Pmin;
    %build polygon
    V1 = [summin(t),Pmax];
    V2 = [vup(t),Pmax];
    V3 = [summax(t),emax(t)];
    V4 = [summax(t),max(L*Pmin,-L*summax(t))];
    V5 = [vdown(t),max(L*Pmin,-L*summax(t))];
    V6 = [summin(t),emin(t)];
    P = Polyhedron([V1;V2;V3;V4;V5;V6]);
    slices = [slices,P];
end

% for t = 1:numslices
% subplot(ceil(numslices/3), 3, t);
% plot(slices(t));
% title(sprintf('T=%d PDF', t));    
%     xlabel(sprintf('State of charge'));
%     ylabel(sprintf('PDF value'));    
% end

