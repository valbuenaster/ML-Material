%% Primero en tres dimensiones, despues en 10 dimensiones
clear,clc,close all

sigma = 0.8;
N = 100;
ww = [1 1 3]';
w = ww/norm(ww);
b = [0.6 0.7 0.8]';

vectP = zeros(3,1);

if(w(3,1) ~= 0)
    vectP(1,1) = 1;
    vectP(2,1) = 1;
    vectP(3,1) = b(3,1) - ((1/w(3,1))*( (w(1,1)*(1-b(1,1))) + (w(2,1)*(1-b(2,1))) ));
else
    if(w(2,1)~=0)
        vectP(1,1) = 1;
        vectP(2,1) = b(2,1) - ((1/w(2,1))*w(1,1)*(1-b(1,1)));
        vectP(3,1) = 0;
    else
        vectP(1,1) = b(1,1);
        vectP(2,1) = 0;
        vectP(3,1) = 0;  
    end
end

vectP = vectP-b;
vectP = vectP./norm(vectP);

vectT = cross(w,vectP);
vectT = vectT/norm(vectT);

Cube = (b + (sigma/2)*[vectP+vectT+w vectP-vectT+w -vectP-vectT+w -vectP+vectT+w vectP+vectT-w vectP-vectT-w -vectP-vectT-w -vectP+vectT-w])';
y = [ones(4,1); -ones(4,1)];

figure,hold on
quiver3(b(1,1),b(2,1),b(3,1),vectP(1,1),vectP(2,1),vectP(3,1),'r')
plot3(0,0,0,b(1,1),b(2,1),b(3,1),'-b*')
quiver3(b(1,1),b(2,1),b(3,1),w(1,1),w(2,1),w(3,1),'b')
quiver3(b(1,1),b(2,1),b(3,1),vectT(1,1),vectT(2,1),vectT(3,1),'g')

plot3(Cube(:,1),Cube(:,2),Cube(:,3),'sr')

indexS = 1 + round(3*(rand(1)));
indexI = 5 + round(3*(rand(1)));

x1 = Cube(indexS,:) + (0.2*sigma*randn(N,3));
x2 = Cube(indexI,:) + (0.2*sigma*randn(N,3));

group1T = plot3(x1(:,1),x1(:,2),x1(:,3),'om','markersize',2);
group2T = plot3(x2(:,1),x2(:,2),x2(:,3),'dc','markersize',2);

axis equal, grid on
set(gcf,'windowstyle','docked')

X = [];
Y = [];
meanSelect = [];%1 for x1, 0 for x2
ii = 0;
flagI = 0;
jj = 0;
flagJ = 0;

while(size(X,1) < (size(x1,1)+size(x2,1)))
    if(rand(1)>=0.5)
        flagI = 1;
        if(ii >= (size(x1,1)))
            jj = jj + 1;
            flagJ = 1;
            flagI = 0;
        else
            ii = ii + 1;
        end
    else
        flagJ = 1;
        if(jj >= (size(x2,1)))
            ii = ii + 1;
            flagI = 1;
            flagJ = 0;
        else
            jj = jj + 1;
        end
    end
    
    if(flagI==1)
        X = [X; x1(ii,:)];
        meanSelect = [meanSelect;1];
        Y = [Y; 1];
    end
    if(flagJ==1)
        X = [X; x2(jj,:)];
        meanSelect = [meanSelect;-1];
        Y = [Y; -1];
    end
    flagI = 0;
    flagJ = 0;
end

%% Train data

XTrain = X;
YTrain = Y;

model=svmtrain(XTrain,YTrain,'kernel_function','linear','boxconstraint',10,'autoscale','false');

% Calculate empirical error
KernelFunction = model.KernelFunction;
SV = model.SupportVectors;
alpha = model.Alpha;
b = model.Bias;
f = (KernelFunction(SV,XTrain)'*alpha) + b;
temp = sign(f);
temp(temp==0)=1;
Y_hat = -temp;

EmpiricalRisk = sum(abs(YTrain - Y_hat))/size(YTrain,1)

empiricalerror = sum((YTrain - Y_hat).^2)/size(YTrain,1)%mean squared error
% Test data

x1 = Cube(indexS,:) + (0.2*sigma*randn(N,3));
x2 = Cube(indexI,:) + (0.2*sigma*randn(N,3));

X = [];
Y = [];
meanSelect = [];%1 for x1, 0 for x2
ii = 0;
flagI = 0;
jj = 0;
flagJ = 0;

while(size(X,1) < (size(x1,1)+size(x2,1)))
    if(rand(1)>=0.5)
        flagI = 1;
        if(ii >= (size(x1,1)))
            jj = jj + 1;
            flagJ = 1;
            flagI = 0;
        else
            ii = ii + 1;
        end
    else
        flagJ = 1;
        if(jj >= (size(x2,1)))
            ii = ii + 1;
            flagI = 1;
            flagJ = 0;
        else
            jj = jj + 1;
        end
    end
    
    if(flagI==1)
        X = [X; x1(ii,:)];
        meanSelect = [meanSelect;1];
        Y = [Y; 1];
    end
    if(flagJ==1)
        X = [X; x2(jj,:)];
        meanSelect = [meanSelect;-1];
        Y = [Y; -1];
    end
    flagI = 0;
    flagJ = 0;
end

XTest = X;
YTest = Y;
group1Ts = plot3(x1(:,1),x1(:,2),x1(:,3),'+m','markersize',2);
group2Ts = plot3(x2(:,1),x2(:,2),x2(:,3),'xc','markersize',2);
f = (KernelFunction(SV,XTest)'*alpha) + b;
temp = sign(f);
temp(temp==0)=1;
Y_hat_test = -temp;

testError = sum((YTest - Y_hat_test).^2)/size(YTest,1)


actualRisk = 0;

for ii=1:size(Y_hat_test,1)
    if(meanSelect(ii,1)==1)
        probb = mvnpdf(X(ii,:),Cube(indexS,:),0.2*sigma*eye(3));
    else
        probb = mvnpdf(X(ii,:),Cube(indexI,:),0.2*sigma*eye(3));
    end
    actualRisk = actualRisk + abs(Y(ii,1)-Y_hat_test(ii,1))*probb;
end
actualRisk
%%
set(group1T,'visible','off')
set(group2T,'visible','off')
set(group1Ts,'visible','on')
set(group2Ts,'visible','on')
%%
set(group1Ts,'visible','off')
set(group2Ts,'visible','off')
set(group1T,'visible','on')
set(group2T,'visible','on')