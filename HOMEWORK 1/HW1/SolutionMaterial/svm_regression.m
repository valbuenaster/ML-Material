%%
clear,clc,close all

gamma = 0.1;
TotalSamples = 100;
y = zeros(2 + TotalSamples,1);
observed = zeros(2 + TotalSamples,1);
X = randn(TotalSamples,1);

x = [0;0;X];

for n = 1:TotalSamples
    y(2 + n,1) =  (0.03*y(1 + n,1)) - (0.01*y(n,1)) + (3*x(2 + n,1)) - (0.5*x(1 + n,1)) + (0.2*x(n,1));
    observed(2 + n,1) = y(2 + n,1) + sqrt(0.1)*randn(1);
end

figure,hold on
plotx = plot(0:TotalSamples-1,x(3:2 + TotalSamples,1),'-b');
ploty = plot(0:TotalSamples-1,y(3:2 + TotalSamples,1),'-r');
plotobs = plot(0:TotalSamples-1,observed(3:2 + TotalSamples,1),'-g');
hh = [plotx ploty plotobs]
legend(hh,'x','y','observed')
axis([0 TotalSamples-1 1.05*min([x;y;observed]) 1.05*max([x;y;observed])])
box on, grid on
set(gcf,'windowstyle','docked')
Y = y(3:2 + TotalSamples,1);

XTrain = [];
XTest = [];
YTrain = [];
YTest = [];

for ii = 1:TotalSamples
    if(rand(1)>0.15)
        XTrain = [XTrain X(ii,1)];
        YTrain = [YTrain Y(ii,1)];
    else
        XTest = [XTest X(ii,1)];
        YTest = [YTest Y(ii,1)];
    end
end
%%
% model=svmtrain(x,y,'kernel_function',precomputedKernel(x,gamma),'boxconstraint',10,'showplot','true','autoscale','false')
model = fitrlinear(XTrain,YTrain)