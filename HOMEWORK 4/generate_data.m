
function [X,B]=generate_data(N,n_elements,phi,A,sigma)
%Inputs: 
%    N: Number of samples 
%    n_elements: Number of antennas (corresponds to dimension D)
%    phi: Vector of L sources
%    A: Vector of amplitudes for the L sources.
%    sigma: Noise standard deviation. 
%
%Outputs
%    X: Matrix of snapshots.
%    B: Matrix of signals (not relevant in this assignment). 
    X=0;
    B=[];
    for i=1:length(A)
        %Produce a set of N complex baseband data fo direction i
        [x,b]=data(N,n_elements,phi(i));
        %Multiply it by its amplitude
        X=X+x*A(i); 
        B=[B b];
    end
    %Add noise
    X=X+sigma*(randn(size(X))+1j*randn(size(X)))/2;

