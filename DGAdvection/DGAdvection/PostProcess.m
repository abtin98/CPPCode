fileID = fopen('data.txt','r');
sizeA = [2 Inf];
formatSpec = '%f %f';
a = 0.5;
finalTime = 2;
A = fscanf(fileID,formatSpec,sizeA);
A = A';
x = A(:,1);
u = A(:,2);
uExact = 0.5*exp(-(x-a*finalTime - 5).^2);

plot(x,u,x,uExact)
legend ('computed solution', 'exact solution')