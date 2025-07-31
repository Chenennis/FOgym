function slices = convert_HFO_to_DFO(FO,COP)
 
 slices1 = FO.slices;
 s1 = size(slices1);
 T = s1(1);
 slices = [];

 for t = 1:T 

     ver = slices1(t).V;
     ver = ver/COP;
     P1 = Polyhedron([ver]);

     slices = [slices,P1];
 end
   
%disp('Done!');

end
