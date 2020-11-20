%%
clear, clc, close all
MS = 9;
FS = 14;

c = [1,1;
     2,1.5;
     2,1;
     3,1.5];
n = 10;
X = [];
% sigma = 0.2;

Delta = 0.1;
sigma = [0.05 1 1.4];

for iter = 1:size(sigma,2)
   ssm = sigma(1,iter);
   X = [];
   for ii = 1:4
      X = [X; ssm*randn(n,2)+repmat(c(ii,:),n,1)];
   end
   Y = [ones(1,2*n) -ones(1,2*n)]';

%    figure
%    plot(X(1:end/2,1),X(1:end/2,2),'+')
%    hold all
%    plot(X(end/2+1:end,1),X(end/2+1:end,2),'o')
%    hold off
%    set(gcf,'windowstyle','docked')
%    set(gca,'fontsize',FS)
%    box on
%    grid on
%    axis square

   figure
   hold on
   model=svmtrain(X,Y,'kernel_function','linear','boxconstraint',100,'showplot','true','autoscale','false')
   set(gcf,'windowstyle','docked')
   fixedAxis = axis

   b = model.Bias;
   SV = model.SupportVectors;
   alpha = model.Alpha;
   KernelFunction = model.KernelFunction;
   
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
   
   hold on
   contour(XX,YY,reshape(ZZ,size(XX)),[0 0],'k','linewidth',2.0);
   contour(XX,YY,reshape(ZZr,size(XX)),[0 0],'r','linewidth',2.0);
   contour(XX,YY,reshape(ZZg,size(XX)),[0 0],'g','linewidth',2.0);

   markers1 = model.FigureHandles{2}(1);
   markers2 = model.FigureHandles{2}(2)
   markers3 = model.FigureHandles{3};
   set(markers1,'markersize',MS)
   set(markers2,'markersize',MS)
   set(markers3,'markersize',MS)
   set(gca,'fontsize',FS)
   
   box on
   grid on
   axis square
   saveas(gcf,strcat('svm_sigma_',num2str(100*ssm)),'epsc')
end





