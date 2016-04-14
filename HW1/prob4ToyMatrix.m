toyT = [0.2 0.3 0.5; 0.4 0.2 0.4; 0.3 0.6 0.1];
toyO = [0.8 0.1 0.1; 0.1 0.4 0.5;0.7 0.2 0.1];
toyp0 = [0.1; 0.2; 0.7];

toyObs = [2 3 1 2 2 1 3];

L = length(toyObs);
dx = size(toyT,1);

curOcol = toyO(:,toyObs(1));
initF = curOcol.*toyp0;

logP = log(sum(initF));

initF = initF./sum(initF);

f = zeros(L,dx);
r = zeros(L,dx);
p = zeros(L,dx);

f(1,:)=initF';

for i = 2:L
   prevF = f(i-1,:);
   curX = (prevF*toyT)';
   curObs = toyO(:,toyObs(i));
   f(i,:)=curObs.*curX;
   logP = logP + log(sum(f(i,:)));
   f(i,:)=f(i,:)./sum(f(i,:));
end

r(L,:) = 1.0;
p(L,:) = r(L,:).*f(L,:);

for t = L-1:-1:1
   prevR = r(t+1,:)';
   curOb = toyO(:,toyObs(t+1));
   curCol = prevR.*curOb;
   r(t,:)=toyT*curCol;
   r(t,:)=r(t,:)./sum(r(t,:));
   p(t,:)=r(t,:).*f(t,:);
   p(t,:)=p(t,:)./sum(p(t,:));
end

logP
p