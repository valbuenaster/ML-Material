%% Primero en tres dimensiones, despues en 10 dimensiones
clear,clc,close all

sigma = 0.45;
N = 50;%when grouping together x1 and x2, there will be 100 samples
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

valuesC = logspace(-1.5,1,100);
ActualRisk = zeros(1,100);
EmpiricalRisk = zeros(1,100);

for iter = 1:size(valuesC,2)
    C = valuesC(1,iter)
    indexS = 1 + round(3*(rand(1)));
    indexI = 5 + round(3*(rand(1)));

    x1a = Cube(indexS,:) + (0.2*sigma*randn(N,10));
    x2a = Cube(indexI,:) + (0.2*sigma*randn(N,10));
    
    [XTrain, YTrain] = organizeData(x1a,x2a);

    model=svmtrain(XTrain,YTrain,'kernel_function','linear','boxconstraint',C,'autoscale','false');

    % Calculate empirical error
    KernelFunction = model.KernelFunction;
    SV = model.SupportVectors;
    alpha = model.Alpha;
    b = model.Bias;
     
    f = (KernelFunction(SV,XTrain)'*alpha) + b;
    Y_hat =  -f;
%     Y_hat = svmclassify(model,XTrain);
    
    EmpiricalRisk(1,iter) = calculateRisk(YTrain,Y_hat);

    indexS = 1 + round(3*(rand(1)));%Take means at random again?
    indexI = 5 + round(3*(rand(1)));%Take means at random again?
    x1b = Cube(indexS,:) + (0.2*sigma*randn(N,10));
    x2b = Cube(indexI,:) + (0.2*sigma*randn(N,10));

    [XTest, YTest] = organizeData(x1b,x2b);

    g = (KernelFunction(SV,XTest)'*alpha) + b;
    Y_hat_test = -g;
%     Y_hat_test = svmclassify(model,XTest);

    ActualRisk(1,iter) = calculateRisk(YTest,Y_hat_test);
    
end
%%
estructuralRisk = abs(ActualRisk - EmpiricalRisk);
figure,hold on
er = plot(valuesC,EmpiricalRisk,'-b');
ar = plot(valuesC,ActualRisk,'-r');
% esr = plot(valuesC,estructuralRisk,'-k');
% hh = [er ar esr];
% legend(hh,'Empirical Risk','Actual Risk','Estructural Risk','location','northwest')
set(gca,'XScale','log')
grid on, box on
% axis([valuesC(1,1) valuesC(1,100) min([ActualRisk EmpiricalRisk estructuralRisk]) max([ActualRisk EmpiricalRisk estructuralRisk])])
% axis([valuesC(1,1) valuesC(1,100) 0 2])
set(gcf,'windowstyle','docked')