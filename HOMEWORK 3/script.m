%%
clear,clc,close all

N = 100
P = [0.8 0.1 0.1;
     0.2 0.5 0.3;
     0.3 0.1 0.6];
 
mu = [1, 2, 3];
sigma = [0.3, 0.3, 0.3];

[x,z]=markovprocess(P,sigma,mu,N)