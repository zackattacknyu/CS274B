thetaX = -1000000:0.01:10;

Atheta = log( exp(thetaX).*(1+exp(1)) + 2);

plot(thetaX,Atheta);