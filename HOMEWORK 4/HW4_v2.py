#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:21:33 2017

@author: Luis Ariel Valbuena Reyes
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
import matplotlib.colors as mplc
import numpy as np
#from svmutil import *
import math

def generate_data(N,n_elements,phi,A,sigma):
#N: Number of samples
#n_elements: Number of antennas (corresponds to dimension D)
#phi: Vector of L sources
#A: Vector of amplitudes for the L sources.
#sigma: Noise standard deviation.   
    B = []
    X = 0
    for ii in range(len(A)):
        x,b = data(N,n_elements,phi[ii,0])
        X = X + x*A[ii,0]
        B.append(b)
        
    X = X + (0.5*sigma*( np.random.normal(len(X)) + ((1j)*np.random.normal(len(X))) ))
    return (X,B)

def data(N,n_elements,phi):
    x = np.matrix( np.zeros((n_elements,N)),dtype=complex )
    temp = np.matrix( np.random.normal(size=N) + ((1j)*( np.sign( np.random.normal(size=N)) )) )
    b = temp.T

    for ii in range(N):
        vector = (1j)*phi*(np.matrix( range(n_elements) ))
        x[:,ii] =  np.exp(vector.T)*b[ii,0]
        
    return (x,b)

def calculateAutoCorrelationMatrix(Data):
    Size = Data.shape[1]
    return (np.dot(Data,Data.H))/Size

def calculate_Power_Spectrum(matrixR,vector_ek):
    matInv = matrixR.I
    divisor = vector_ek[0,0]
    ee_k = vector_ek/divisor
    
    return 1.0/((ee_k.H)*matInv*ee_k)

def calculate_MUSIC(matrixR,vector_ek):
    divisor = vector_ek[0,0]
    ee_k = vector_ek/divisor
    
    return 1.0/((ee_k.H)*matrixR*ee_k)

def calculate_Power_SpectrumSignal(matrixS,vector_ek):
    divisor = vector_ek[0,0]
    ee_k = vector_ek/divisor
    return (ee_k.H)*matrixS*ee_k


def separate_Signal_Noise(eigenvalues,eigenvectors):
    N = eigenvalues.shape[0]
    for ii in range(N-1):
        temp = np.abs(eigenvalues[ii]/eigenvalues[ii+1])
        if (temp > 10000):
            break
    eigValS = eigenvalues[0:ii+1]
    eigValN = eigenvalues[ii+1:N]
    eigVecS = eigenvectors[:,0:ii+1]
    eigVecN = eigenvectors[:,ii+1:N]
    
    return (eigValS,eigVecS,eigValN,eigVecN)

def gramschmidt(W):
    V = np.matrix(W,dtype=complex)#,dtype=complex
#    print "\nV"
#    print V
    Columns = V.shape[1]
    V[:,0] = V[:,0]/np.sqrt(np.dot(V[:,0].T,V[:,0]))
#    print "V[:,0]"
#    print V[:,0]
    for ii in range(1,Columns):
#        print "ii " + str(ii)
        for jj in range(ii):
#            print "jj " + str(jj)
            V[:,ii] = V[:,ii] - np.multiply( np.dot(V[:,ii].H,V[:,jj]), V[:,jj])
#        print np.dot(V[:,ii].T,V[:,ii])
        V[:,ii] = V[:,ii]/np.sqrt(np.dot(V[:,ii].H,V[:,ii]))
    return V    


if __name__ == "__main__":
    Samples = 100
    D = 10
    sigma = 3.0
    phi_1 = (math.pi)/2
    phi_2 = (math.pi)
    phi_3 = 3*(math.pi)/2
    vector_phi = np.matrix([[phi_1],[phi_2],[phi_3]])
    vector_Amplitude = np.matrix([[5.2],[3.5],[2.7]])#I just made this up
    
    #Generate 100 samples of the data
    X, B = generate_data(Samples,D,vector_phi,vector_Amplitude,sigma)
    
    #Compute the autocorrelation matrix of the data
    R = calculateAutoCorrelationMatrix(X)
    
    #Compute all eigenvectors
    EigenValues, V = np.linalg.eig(R)
    eigValsSignal, eigVectSignal, eigValsNoise, eigVectNoise = separate_Signal_Noise(EigenValues,V)
    
    #Compute the inverse of the power spectrum of the noise eigenvectors.
    PowerSpectrum_eigVectNoise = np.matrix(np.zeros((eigVectNoise.shape[1],1)),dtype=complex)
    for ii in range(eigVectNoise.shape[1]):
        tempval = calculate_Power_Spectrum(R,eigVectNoise[:,ii])
        PowerSpectrum_eigVectNoise[ii,0] = tempval.tolist()[0][0]

    #Use this expression to compute the MUSIC algorithm.

    #Represent the MUSIC spectrum in dB.
#    Vn = np.matrix(np.zeros(( D, D)),dtype=complex)
#    for jj in range(eigValsSignal.shape[0],D):
#        Vn[jj,jj] = eigValsNoise[jj-eigValsSignal.shape[0]]
    
    angles = np.arange(0,2*math.pi,math.pi/50).tolist()
    SSpectrum = []
    VnVnH = np.dot(eigVectNoise,eigVectNoise.H)
    for phi in angles:
        phi_data = data(1,D,phi)[0]
        tempval = calculate_MUSIC(VnVnH,phi_data)
        SSpectrum.append( tempval.tolist()[0][0] )
    
    output_dB = []    
    for element in SSpectrum:
        output_dB.append(20*np.log10(np.abs(element)))
        
    anglesdisplay = np.arange(0,2*math.pi,math.pi/4).tolist()
    pl.figure()
#    pl.plot(angles,output_dB,'-bs',markeredgecolor = 'k')
    pl.plot(angles,output_dB,'b')
    pl.grid()
    pl.axis([0,2*math.pi,np.min(output_dB)-10,np.max(output_dB)+10])
    pl.xticks(anglesdisplay)
    pl.title("MUSIC NOISE EIGENVECTORS")
    pl.ylabel('dB')
    pl.savefig('Drawings/'+'MUSICNOISEEIGENVECTORS.eps', bbox_inches='tight')
    pl.show()
        
    #Compute the three first eigenvectors V s of the autocorrelation, which
    #are supposed to correspond to the signal eigenvectors. Since
    #                     VV = V s V s + V n V n = I
    #derive an expression that is equivalent to the inverse of the power 
    #spectrum of the noise eigenvalues.
#    AAA = np.dot(eigVectNoise,eigVectNoise.H) + np.dot(eigVectSignal,eigVectSignal.H)

    EquivVsVshmatrix = np.matrix( np.identity(D)- np.dot(eigVectSignal,eigVectSignal.H),dtype=complex)
    SSpectrumSig = []
    for phi in angles:
        phi_data = data(1,D,phi)[0]
        tempval = calculate_MUSIC(EquivVsVshmatrix,phi_data)
        SSpectrumSig.append( tempval.tolist()[0][0] )
     
    #Use this expression to compute the MUSIC algorithm.

    #Represent the MUSIC spectrum in dB and check that both graps are
    #equal
    
    output_dBSig = []    
    for element in SSpectrumSig:
        output_dBSig.append(20*np.log10(np.abs(element)))
          
    pl.figure()
#    pl.plot(angles,output_dB,'-ro',markeredgecolor = 'k')
    pl.plot(angles,output_dB,'r')
    pl.grid()
    pl.axis([0,2*math.pi,np.min(output_dB)-10,np.max(output_dB)+10])
    pl.xticks(anglesdisplay)
    pl.title("MUSIC SIGNAL EIGENVECTORS")
    pl.ylabel('dB')
    pl.savefig('Drawings/'+'MUSICSIGNALEIGENVECTORS.eps', bbox_inches='tight')
    pl.show() 
    
#    Testmatrix = np.matrix([[1.0,2.0,3.0],[4.0,-5.0,6.0],[7.0,8.0,9.0]],dtype=complex)
#    orthoTestmatrix = gramschmidt(Testmatrix)
#    Prueba = np.dot(orthoTestmatrix,orthoTestmatrix.H)
#    print "Prueba"
#    print Prueba

    #PPCA
#    W = np.matrix(np.zeros((D,D)),dtype=complex) 
#    W = np.matrix(np.identity(D))
#    II = np.matrix(np.identity(D))
#             
#    print "\nW\n"
#    print W
#    ii = 0
#    arrayLambdA = np.zeros((D,1),dtype=complex)
#    
#    while(ii<30):
#        WW = np.dot(W.H,W)
#        Z = (WW.I)*(W.H)*X 
##        print "\nZ\n"
##        print Z
#        matAux = np.dot(np.dot(X,Z.H),np.dot(Z,Z.H))
#        W = matAux.I + np.dot(II - R,W)
#        ii = ii +1
#        W = gramschmidt(W)         
#        print "\nii = " + str(ii)
#        LambdA = np.dot(np.dot(W,R),W)
#        for kk in range(D):
#            arrayLambdA[kk] = LambdA[kk,kk]
#            
#    print "\nLambdA\n"
#    print arrayLambdA
#    print "\nEigenValues\n" 
#    print EigenValues   
    
    
#    W = np.matrix(np.identity(D))
#    II = np.matrix(np.identity(D))
#             
#    print "\nW\n"
#    print W
#    ii = 0
#    arrayLambdA = np.zeros((D,1),dtype=complex)
#    
#    while(ii<1):
#        WW = np.dot(W.H,W)
#        Z = np.dot(WW.I,np.dot(W.H,X)) 
##        print "\nZ\n"
##        print Z
#        ttemp = np.dot(Z,Z.H)
#        Wp = np.dot(np.dot(X,Z.H),ttemp.I) + np.dot(II - R,W)
#        ii = ii +1
#        Wpp = np.dot(Wp,np.dot(Wp.H,np.dot(R,Wp)))
#        W = gramschmidt(Wpp)         
#        print "\nii = " + str(ii)
#        LambdA = np.dot(np.dot(W.H,R),W)
#        for kk in range(D):
#            arrayLambdA[kk] = LambdA[kk,kk]
#            
#    print "\narrayLambdA\n"
#    print arrayLambdA
#    print "\nEigenValues\n" 
#    print EigenValues
    
    
    W = np.matrix(np.identity(D))
    II = np.matrix(np.identity(D))
             
    print "\nW\n"
    print W
    ii = 0
    arrayLambdA = np.zeros((D,1),dtype=complex)
    
    while(ii<4000):
        WW = W.H*W
        Z =(WW.I)*(W.H*X)
#        print "\nZ\n"
#        print Z
        ttemp = Z*Z.H
        Wp = ((X*Z.H)*ttemp.I)
        Wpp = Wp*Wp.H*R*Wp
        ii = ii +1
        W = gramschmidt(Wpp)      
#        print "\nii = " + str(ii)
        LambdA = ((W.H*R)*W)
        for kk in range(D):
            arrayLambdA[kk] = LambdA[kk,kk]
            
    print "\narrayLambdA\n"
    print arrayLambdA
    print "\nEigenValueseigValsSignal\n" 
    print eigValsSignal
    print "\nEigenValueseigValsNoise\n" 
    print eigValsNoise