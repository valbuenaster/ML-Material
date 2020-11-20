%% Primero en tres dimensiones, despues en 10 dimensiones
clear,clc,close all

sigma = 1;

C = 1;
N = 10:10:500;
Trials = 300;

ActualRisk = zeros(Trials,size(N,2));
EmpiricalRisk = zeros(Trials,size(N,2));
averagedEmpiricalRisk = zeros(1,size(N,2));
averagedActualRisk = zeros(1,size(N,2));

for iteration = 1:Trials
    for iter = 1:size(N,2)

        [XTrain,YTrain]=data(N(1,iter),sigma);
        model = svmtrain(YTrain,XTrain,'-s 0 -t 0 -c 1');
        [Y_hat accuracy prob_estimates] = svmpredict(YTrain,XTrain,model);
%         EmpiricalRisk(iteration,iter) = calculateRisk(YTrain,Y_hat);
        EmpiricalRisk(iteration,iter) = 1 - prob_estimates(1)/100;

        [XTest,YTest]=data(N(1,iter),sigma);
        [Y_hat_test accuracy prob_estimates] = svmpredict(YTest,XTest,model);
%         ActualRisk(iteration,iter) = calculateRisk(YTest,Y_hat_test);
        ActualRisk(iteration,iter) = 1 - prob_estimates(1)/100;

    end
end

for kk = 1:size(N,2)
    averagedEmpiricalRisk(1,kk) = sum(EmpiricalRisk(:,kk))/Trials;
    averagedActualRisk(1,kk) = sum(ActualRisk(:,kk))/Trials;
end


% estructuralRisk = abs(ActualRisk - EmpiricalRisk);
figure,hold on
erN = plot(N,averagedEmpiricalRisk,'-b');
arN = plot(N,averagedActualRisk,'-r');
% srN = plot(N,averagedActualRisk-averagedEmpiricalRisk,'-g');
hh = [erN arN];
legend(hh,'Empirical Risk','Actual Risk','Location','Northeast')
grid on, box on
axis([N(1,1) N(1,size(N,2)) min([averagedActualRisk averagedEmpiricalRisk]) max([averagedActualRisk averagedEmpiricalRisk])])
% set(gca,'YScale','log')
set(gca,'fontsize',14)
set(gcf,'windowstyle','docked')
saveas(gcf,'RiskVSNElements','epsc')