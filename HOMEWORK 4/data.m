function [x,b]=data(N,n_elements,phi)
    %Simulate data in a N element array
    x=zeros(n_elements,N); %Snapshots (row vectors)
    b=sign(randn(N,1)) + 1j*sign(randn(N,1)); %Data
    for i=1:N
        x(:,i)=exp(1j*phi*(0:n_elements-1)')*b(i);
    end