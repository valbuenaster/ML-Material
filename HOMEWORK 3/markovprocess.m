function [x,z]=markovprocess(P,sigma,mu,N)
P
p=cumsum(P,2)
zact=ceil(rand*length(mu));
z=[];
for i=1:N
    a=rand;
    zact=[min(find(p(zact,:)>a))]
    z=[z zact];
end
z
x=randn(size(z)).*sigma(z)+mu(z);
x=x';