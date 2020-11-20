%% Primero en tres dimensiones, despues en 10 dimensiones
clear,clc,close all

sigma = 0.3;
ww = ones(10,1);
w = ww/norm(ww);
offset = [0.6 0.7 0.8 0.35 0.15 0.9 0.2 0.44 0.5 1]';
% offset = zeros(10,1);

vectP = [ones(5,1); -ones(5,1)];
vectP = vectP - dot(vectP,w)*w;
vectP = vectP./norm(vectP);

vectT = [1 -1  1  -1  1 -1  1 -1  1 -1]';
vectT = vectT - dot(vectT,w)*w - dot(vectT,vectP)*vectP;
vectT = vectT/norm(vectT);

Cube = (offset + (sigma/2)*[vectP+vectT+w vectP-vectT+w -vectP-vectT+w -vectP+vectT+w vectP+vectT-w vectP-vectT-w -vectP-vectT-w -vectP+vectT-w])';
y = [ones(4,1); -ones(4,1)];

C = 1;
N = 10:10:500;
ActualRisk = zeros(1,size(N,2));
EmpiricalRisk = zeros(1,size(N,2));

for iter = 1:size(N,2)
    indexS = 1 + round(3*(rand(1)));
    indexI = 5 + round(3*(rand(1)));

    x1a = Cube(indexS,:) + (0.5*sigma*randn(N(1,iter),10));
    x2a = Cube(indexI,:) + (0.5*sigma*randn(N(1,iter),10));
    
    [XTrain, YTrain] = organizeData(x1a,x2a);

    model=svmtrain(XTrain,YTrain,'kernel_function','linear','boxconstraint',C,'autoscale','false');

    % Calculate empirical error
    KernelFunction = model.KernelFunction;
    SV = model.SupportVectors;
    alpha = model.Alpha;
    b = model.Bias;
    
    
    f = (KernelFunction(SV,XTrain)'*alpha) + b;
%     Y_hat =  -f;
    Y_hat = svmclassify(model,XTrain);
    
    
    EmpiricalRisk(1,iter) = calculateRisk(YTrain,Y_hat);

    indexS = 1 + round(3*(rand(1)));%Take means at random again?
    indexI = 5 + round(3*(rand(1)));%Take means at random again?
    x1b = Cube(indexS,:) + (0.5*sigma*randn(N(1,iter),10));
    x2b = Cube(indexI,:) + (0.5*sigma*randn(N(1,iter),10));

    [XTest, YTest] = organizeData(x1b,x2b);

    g = (KernelFunction(SV,XTest)'*alpha) + b;
%     Y_hat_test = -g;
    Y_hat_test = svmclassify(model,XTest);

    ActualRisk(1,iter) = calculateRisk(YTest,Y_hat_test);
    
end

estructuralRisk = abs(ActualRisk - EmpiricalRisk);
figure,hold on
erN = plot(N,EmpiricalRisk,'-b');
arN = plot(N,ActualRisk,'-r');
hh = [erN arN];
legend(hh,'Empirical Risk','Actual Risk','Location','Northwest')
grid on, box on
axis([N(1,1) N(1,size(N,2)) min([ActualRisk EmpiricalRisk]) max([ActualRisk EmpiricalRisk])])
set(gcf,'windowstyle','docked')