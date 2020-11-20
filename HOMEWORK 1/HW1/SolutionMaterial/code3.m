%% Primero en tres dimensiones, despues en 10 dimensiones
clear,clc,close all

sigma = 0.5;
N = 100;
ww = [2 1 1]';
w = ww/norm(ww);
b = [0.6 0.7 00.8]';

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

Cube = (b + (sigma/2)*[vectP+vectT+w vectP-vectT+w -vectP-vectT+w -vectP+vectT+w vectP+vectT-w vectP-vectT-w -vectP-vectT-w -vectP+vectT-w])'
y = [ones(4,1); -ones(4,1)]

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

plot3(x1(:,1),x1(:,2),x1(:,3),'om','markersize',2)
plot3(x2(:,1),x2(:,2),x2(:,3),'dc','markersize',2)

axis equal, grid on
set(gcf,'windowstyle','docked')

X = []
Y = []
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
        Y = [Y; 1];
    end
    if(flagJ==1)
        X = [X; x2(jj,:)];
        Y = [Y; -1];
    end
    flagI = 0;
    flagJ = 0;
end

% Train and test data

XTrain = [];
XTest = [];
YTrain = [];
YTest = [];


for ii=1:size(X,1)
    if(mod(ii,7)~=0)
        XTrain = [XTrain; X(ii,:)];
        YTrain = [YTrain; Y(ii,:)];
    else
        XTest = [XTest; X(ii,:)];
        YTest = [YTest; Y(ii,:)];
    end
end

model=svmtrain(XTrain,YTrain,'kernel_function','linear','boxconstraint',100,'showplot','true','autoscale','false')

%%Calculate empirical error
KernelFunction = model.KernelFunction;
SV = model.SupportVectors;
alpha = model.Alpha;
b = model.Bias;
f = (KernelFunction(SV,XTrain)'*alpha) + b;
temp = sign(f);
temp(temp==0)=1;
Y_hat = -temp;

difference = YTrain - Y_hat