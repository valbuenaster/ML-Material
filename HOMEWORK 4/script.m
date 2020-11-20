%%
clear,clc,close all

N = 100
D = 10
phi = pi*[0.5 1 1.5]'
sigma = 3
Amplitude = [1.87 2.31 1.95]' % randomly chosen

[X,B]=generate_data(N,D,phi,Amplitude,sigma);