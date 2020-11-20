%%
clear, clc, close all

MS = 10;

c = [1,1;
     2,1.5;
     2,1;
     3,1.5];
n = 10;
X = [];
sigma = 0.2;

for ii = 1:4
    X = [X; sigma*randn(n,2)+repmat(c(ii,:),n,1)];
end

Y = [ones(1,2*n) -ones(1,2*n)]';
figa = figure;
plot(X(1:end/2,1),X(1:end/2,2),'g*','markersize',6)
hold all
plot(X(end/2+1:end,1),X(end/2+1:end,2),'r+','markersize',6)
hold off
set(gcf,'windowstyle','docked')

figb = figure;
hold on
model=svmtrain(X,Y,'kernel_function','linear','boxconstraint',100,'showplot','true','autoscale','false')
set(gcf,'windowstyle','docked')
fixedAxis = axis;
box on
grid on
axis square

w1 = (((X')*X)^-1)*(X')*Y
b = model.Bias;
SV = model.SupportVectors;
alpha = model.Alpha;
SVindices = model.SupportVectorIndices;
KernelFunction = model.KernelFunction;

markers1 = model.FigureHandles{2}(1);
markers2 = model.FigureHandles{2}(2)
markers3 = model.FigureHandles{3};
set(markers1,'markersize',MS)
set(markers2,'markersize',MS)
set(markers3,'markersize',MS)
set(gca,'fontsize',14)
saveas(gcf,'NormalExecutionE','epsc')
%%
% result = X*w1;
% figure(figa),hold on
% for ii=1:size(result,1)
%     if(result(ii,1)<0.0)
%         plot(X(ii,1),X(ii,2),'ro','markersize',10)
%     else
%         plot(X(ii,1),X(ii,2),'bs','markersize',10)
%     end 
% end
% KK = 2;
% plot([0 KK*w1(2)],[0 -KK*w1(1)],'k')
%%
[U,S,V] = svd(X,'econ');
myAlpha = (U*((S^2)^-1)*U')*Y;
w2 = X'*myAlpha

Y_hat = X*w2;
Y_masked = zeros(size(Y_hat));
figure(figa),hold on
for ii=1:size(Y_hat,1)
    if(Y_hat(ii,1)>0.0)
        lr = plot(X(ii,1),X(ii,2),'ro','markersize',10)
        Y_masked(ii,1) = -1;
    else
        lb = plot(X(ii,1),X(ii,2),'bs','markersize',10)
        Y_masked(ii,1) = 1;
    end 
end
KK = 2;

hh = [lb lr];
legend(hh,'-1','1')

axis(fixedAxis)
box on

PointsPlane = [X Y_hat Y_masked];%This is for display purposes

%Calculate the plane of my machine
[XX YY] = meshgrid(linspace(fixedAxis(1),fixedAxis(2)), linspace(fixedAxis(3),fixedAxis(4)));
[xxlim yylim] = size(XX);

ZZ = zeros(xxlim,yylim);

for ii = 1:xxlim
    for jj = 1:yylim
        ZZ(ii,jj) = dot([XX(ii,jj) YY(ii,jj) 0],[w2;0]);
    end
end

figure(figa),hold on
contour(XX,YY,ZZ,[0 0],'c','linewidth',2.0);% my machine

%% From svmtrain data
[XX YY] = meshgrid(linspace(fixedAxis(1),fixedAxis(2)), linspace(fixedAxis(3),fixedAxis(4)));
[xxlim yylim] = size(XX);

f = (KernelFunction(SV,[XX(:) YY(:)])'*alpha) + b;
fr = (KernelFunction(SV,[XX(:) YY(:)])'*alpha) + b - 1;
fg = (KernelFunction(SV,[XX(:) YY(:)])'*alpha) + b + 1;
temp = sign(f);
temp(temp==0)=1;
ZZ = temp;
temp = sign(fr);
temp(temp==0)=1;
ZZr = temp;
temp = sign(fg);
temp(temp==0)=1;
ZZg = temp;

figure(figb),hold on
contour(XX,YY,reshape(ZZ,size(XX)),[0 0],'c');
contour(XX,YY,reshape(ZZr,size(XX)),[0 0],'r');
contour(XX,YY,reshape(ZZg,size(XX)),[0 0],'g');

figure, hold on
surf(XX,YY,reshape(f,size(XX)))

figure(figa),hold on
contour(XX,YY,reshape(ZZ,size(XX)),[0 0],'-k*','linewidth',2.0);% machine svmtrain

box on
grid on
axis square
set(gca,'fontsize',14)
saveas(gcf,'ComparizonE','epsc')