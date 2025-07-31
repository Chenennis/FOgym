function AF = assignment_flexibility_granularity(FO,gr,TF)

if ~exist('TF','var')
    TF = 0;
end

slices = FO.slices;
length = size(slices);
k = length(1);
old = [0;1];

for t = 1:k
   slice = slices(t);
   vertices = (slice.V)*gr;
   polix = vertices(:,1);
   poliy = vertices(:,2);
   maxlength = max(vertices(:,1)); %find length of FO slice
   minlength = min(vertices(:,1));
   maxheight = max(vertices(:,2));
   minheight = min(vertices(:,2));
   new1 = [ceil(minheight+minlength):floor(maxlength+maxheight)];
   subtr = ceil(minheight+minlength)-1;
   siio = size(old);
   siio2 = siio(2);
   siin = size(new1);
   siin2 = siin(2);
   newocc = repelem(0,siin2);
   new = [new1;newocc];
   mh = ceil(minheight);
   Mh = floor(maxheight);
   for j = 1:siio2
       for j1 = 0:(Mh-mh) %create loop for new values
           testval = old(1,j)+j1+mh;%create the sum of old and new value
           testres = inpolygon(old(1,j),mh+j1,polix,poliy);%see if it belongs to the polygon
           if testres == 1
               new(2,testval-subtr) = new(2,testval-subtr) + old(2,j);%add the occurrencies count in the right place
           end
       end
   end
   old = new;
end
AF = sum(new(2,:))/(gr)^k * (TF+1);
end