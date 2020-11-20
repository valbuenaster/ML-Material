%%
clear,clc,close all
c=1;
NN=10:500;
Remp=zeros(1000,190);
R=Remp;
for i=1:100
    for j=1:length(NN)

        [X,Y]=data(NN(j),1);

        model=svmtrain(Y,X,'-s 0 -t 0 -c 1');
        [y_,a,p]=svmpredict(Y,X,model);
        Remp(i,j)=1-a(1)/100;

        
        [X,Y]=data(1000,1);
        [y_,a,p]=svmpredict(Y,X,model);
        R(i,j)=1-a(1)/100;
    end
    figure(1)
    plot(NN,mean(Remp(1:i,:)),'r')
    hold on
    plot(NN,mean(R(1:i,:)))

 
    hold off
    drawnow
end

 
keyboard

%  
% function [X,Y]=data(N,sigma)
% w=ones(1,10)/sqrt(10); 
% w1=w.*[ 1  1  1  1  1 -1 -1 -1 -1 -1];
% w2=w.*[-1 -1  0  1  1 -1 -1 0  1  1];
% w2=w2/norm(w2);
% 
%  
% x(1,:)=zeros(1,10);
% x(2,:)=x(1,:)+sigma*w1;
% x(3,:)=x(1,:)+sigma*w2;
% x(4,:)=x(3,:)+sigma*w1;
% X1=x+sigma*repmat(w,4,1)/2;
% X2=x-sigma*repmat(w,4,1)/2;
% X1=repmat(X1,2*N,1);
% X2=repmat(X2,2*N,1);
% X=[X1;X2];
% Y=[ones(4*2*N,1);-ones(4*2*N,1)];
% Z=randperm(8*2*N);
% Z=Z(1:N);
% X=X(Z,:)+0.4*sigma*randn(size(X(Z,:)));
% Y=Y(Z);