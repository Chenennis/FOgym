function Constraints = build_dfo_slices_closein(mCtr, eVars)
 T = size(eVars, 2);
 disp(value(T));
 
 if T ~= 8
     disp('Wrong time')
 end
 % Add additional constraints
 cCtr = mCtr;
 eTotal = sdpvar(1, T);
 eLeft = sdpvar(1, T);
  
 for t = 1:T-1
     cCtr = [cCtr; eTotal(t) == sum(eVars(1:t)); eLeft(t) == sum(eVars(t+1:T))];
 end

% Display message
fprintf('Building CloseInDFO slices.');

ConstraintsA = [];
Constraintsb = [];

P1 = Polyhedron(lmi(projection(cCtr, [eTotal(4), eLeft(4)])));
P1 = Polyhedron('A', full(P1.A), 'b', full(P1.b), 'Ae', full(P1.Ae), 'be', full(P1.be));
P1A = P1.A;
P1b = P1.b;
siz = size(P1.A);
for k = 1:siz(1)
    ConstraintsA = [ConstraintsA;[repelem(P1A(k,1),4),repelem(P1A(k,2),4)]];
end
Constraintsb = [Constraintsb; P1b];

%Idea: sviluppa i constraint in maniera più capillare, in modo tale che
%nelle variabili ci siano anche i constraint introdotti in precedenza.
%Possibilmente sarebbe meglio mettere insieme constraints "affini", cioè
%simili tra loro. 

   
disp('Done!');

end