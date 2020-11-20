%% 
clear,clc,close all

sigma = 0.15;
N = 1000;
Nn = 400;
Realizations = 800;

% valuesC = logspace(-1.0706,7,Nn);
valuesC = logspace(-2,7,Nn);
ActualRisk = zeros(1,Nn);
EmpiricalRisk = zeros(1,Nn);
otherEmpiricalRisk = zeros(1,size(valuesC,2));

% Take many iterations
MatrixIterationsEmpirical = zeros(Realizations,Nn);
MatrixIterationsActual = zeros(Realizations,Nn);

for iteration = 1:Realizations
    for iter = 1:Nn
        C = valuesC(1,iter);
        Cadena = ['-s 0 -t 0 -c ',num2str(C)];
        [XTrain,YTrain]=data(N,sigma);
        [XTest,YTest]=data(N,sigma);
    %     model = svmtrain(XTrain,YTrain,'kernel_function','polynomial','boxconstraint',C,'polyorder',4,'autoscale','false');
        model = svmtrain(YTrain,XTrain,Cadena);
    %     KernelFunction = model.KernelFunction;
    %     SV = model.SupportVectors;
    %     alpha = model.Alpha;
    %     b = model.Bias;

    %     f = (KernelFunction(SV,XTrain)'*alpha) + b;
    %     Y_hat =  -f;
        [Y_hat accuracy prob_estimates] = svmpredict(YTrain,XTrain,model);

        EmpiricalRisk(1,iter) = calculateRisk(YTrain,Y_hat);
        MatrixIterationsEmpirical(iteration,iter) = EmpiricalRisk(1,iter);

    %     temp1 = YTrain-Y_hat;
    %     for iii=1:size(temp1,1)
    %         otherEmpiricalRisk(1,iter) = otherEmpiricalRisk(1,iter) + abs(temp1(iii,1));
    %     end
    % otherEmpiricalRisk(1,iter) = otherEmpiricalRisk(1,iter)/(2*size(temp1,1));


    %     g = (KernelFunction(SV,XTest)'*alpha) + b;
    %     Y_hat_test = -g;
        [Y_hat_test accuracy prob_estimates]= svmpredict(YTest,XTest,model);

        ActualRisk(1,iter) = calculateRisk(YTest,Y_hat_test);
        MatrixIterationsActual(iteration,iter) = ActualRisk(1,iter);
        aaa = [iteration iter]
    end
end
%%
averagedEmpiricalRisk = zeros(1,Nn);
averagedActualRisk = zeros(1,Nn);
for kk=1:Nn
    averagedEmpiricalRisk(1,kk) = sum(MatrixIterationsEmpirical(:,kk))/Nn;
    averagedActualRisk(1,kk) = sum(MatrixIterationsActual(:,kk))/Nn;
end
%%
estructuralRisk = abs(averagedActualRisk - averagedEmpiricalRisk);
figure,hold on
er = plot(valuesC,averagedEmpiricalRisk,'-b');
ar = plot(valuesC,averagedActualRisk,'-r');
% er = plot(valuesC,EmpiricalRisk,'-b');
% ar = plot(valuesC,ActualRisk,'-r');
esr = plot(valuesC,estructuralRisk,'-k');
hh = [er ar esr];
legend(hh,'Empirical Risk','Actual Risk','Estructural Risk','location','northeast')
set(gca,'XScale','log')
grid on, box on
% axis([valuesC(1,1) valuesC(1,100) min([ActualRisk EmpiricalRisk estructuralRisk]) max([ActualRisk EmpiricalRisk estructuralRisk])])
% axis([valuesC(1,1) valuesC(1,100) 0 2])
set(gca,'fontsize',14)
set(gcf,'windowstyle','docked')
saveas(gcf,'RisksVSValueC','epsc')
%%
axis([0.035 10000000 0 0.023])
% axis([0.035 10000000 0 0.023])
saveas(gcf,'RisksVSValueC2','epsc')
save Experiment3 valuesC averagedEmpiricalRisk averagedActualRisk estructuralRisk MatrixIterationsEmpirical MatrixIterationsActual