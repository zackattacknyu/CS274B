f12 = ones(3);
f13 = ones(3);
f23 = ones(3);

prod1 = f13*(f23');
p12Init = f12.*prod1;
p12Init = p12Init./sum(p12Init(:));

phat12 = [0.249 0.002 0.311;...
    0.017 0.024 0.015;...
    0.029 0.348 0.000];

newF12 = f12.*phat12./p12Init

p13Init = f13.*(newF12*f23);
p13Init = p13Init./sum(p13Init(:))

phat13 = [0.136 0.118 0.309;...
    0.003 0.029 0.025;
    0.0348 0.014 0.015];

newF13 = f13.*phat13./p13Init