thetaX = -10:0.01:10;

Atheta = log( exp(thetaX).*(1+exp(1)) + 2);

figure
plot(thetaX,Atheta);
xlabel('\theta_x');
ylabel('A(\theta)');