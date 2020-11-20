#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 22:50:44 2017

@author: Luis Ariel Valbuena Reyes
"""
from homewok2_StoppingCriterion import *
import numpy as np
import math
import matplotlib.pyplot as pl

def create_10D_Data(N,sigma):
    w = (1/math.sqrt(10))*np.matrix([1,1,1,1,1,1,1,1,1,1])
    w1 = (1/math.sqrt(10))*np.matrix([1,1,1,1,1,-1,-1,-1,-1,-1])
    temp = np.matrix([-1,-1,0,1,1,-1,-1,0,1,1])
    w2 = temp/math.sqrt(np.dot(temp,temp.T))
    
    x = np.zeros((4,10))
    x[1][:] = x[0][:] + sigma*w1
    x[2][:] = x[0][:] + sigma*w2
    x[3][:] = x[2][:] + sigma*w1
    
    X1 = x + (sigma/2)*np.tile(w,(4,1))
    X2 = x - (sigma/2)*np.tile(w,(4,1))
    X1 = np.tile(X1,(2*N,1))
    X2 = np.tile(X2,(2*N,1))
    X = np.concatenate((X1,X2))
    Y = np.concatenate((np.ones((4*2*N,1)),-np.ones((4*2*N,1))))
    
    arr = np.arange(8*2*N)
    np.random.shuffle(arr)
    
    arr = arr[0:N]
    
    SSS = X[arr][:] 
    dimX = len(SSS)
    dimY = SSS[0].size
    
    NewX = SSS + sigma*np.random.normal(0.0,0.2,(dimX,dimY))
    NewY = Y[arr]

    return (NewX,NewY)

def sortData(X,randomValue):
    indexCounter = 0
    TempMatrix = np.matrix([[0, 0]])
    dimX,dimY = X.shape
#    print 'dimY ' + str(dimY)
    for ii in range(dimY):
        x = X[:,ii]
        temp = x - randomValue
        norm = math.sqrt(temp.T*temp)
        TempMatrix = np.concatenate((TempMatrix,np.matrix([norm,indexCounter])))
        indexCounter = indexCounter + 1
        
    TempMatrixM = np.array(TempMatrix[1:dimY+1,:])
    TempMatrixM.view('i8,i8').sort(order = ['f0'],axis=0)
    Xs = np.matrix(np.zeros((dimX,1)))

    for index in TempMatrixM[:,1]:
        ii = int(index)
        Xt = X[:,index]
        Xs = np.concatenate((Xs,Xt), axis=1) 
#    print "Xs "
#    print Xs.shape                  
    return (Xs[:,1:dimY+1],TempMatrixM)
    
def assignProbability(dists):
    SumDist = np.sum(dists)
    vectPro = []
    for d in dists:
        vectPro.append(d/SumDist)
    return vectPro

def findIndexMax(ProbabilityList,EstFeatures):
    listSlopes = []
    for jj in range(1,len(Probability)):
        listSlopes.append(Probability[jj] - Probability[jj-1])
    
    returnIndices = []
    for ii in range(EstFeatures-1):
        MaxSlope = max(listSlopes)
        indices = [i for i, j in enumerate(listSlopes) if j == MaxSlope]
        for elem in indices:
            listSlopes[elem] = 0
            returnIndices.append(elem+1)
    try:
        aa = indices.index(0)
        indices.pop(aa)
        return returnIndices
    except ValueError:
        return returnIndices

def growing(mu,CV,scaletol):
    U,S,V = np.linalg.svd(CV)
    print "singular values "
    norm2 = max(S)
    majorAxis = U[:,0]
#    print majorAxis
    scaledEigenvalues = []
    Flag = 0
    for ss in S:
        print ss/norm2
        scaledEigenvalues.append(ss/norm2)
        if(scaledEigenvalues[-1]<scaletol):
            Flag = 1
    if(Flag == 1):
        new_mu_1 = np.matrix([[float(media_X[0] + math.sqrt(S[0])*U[0,0])],[float(media_X[1] + math.sqrt(S[0])*U[1,0])]])
        new_mu_2 = np.matrix([[float(media_X[0] - math.sqrt(S[0])*U[0,0])],[float(media_X[1] - math.sqrt(S[0])*U[1,0])]])
        new_CV_1 = np.matrix([[1, 0],[0, 1]])
        new_CV_2 = np.matrix([[1, 0],[0, 1]])
        return ([new_mu_1, new_mu_2],[new_CV_1, new_CV_2])
    else:
        return -1
        
    return (U,S)
    

def prunning(mu_1,mu_2,CV_1,CV_2,tol_mu,tol_CV):
    difference_mu = mu_1 - mu_2
    difference_CV = CV_1 - CV_2

    norm_diff = math.sqrt(np.dot(difference_mu.T,difference_mu))
    U, S, V = np.linalg.svd(difference_CV)
    norm_diff_CV = S[0]
#    print "norm_diff " + str(norm_diff) + "\n"   
#    print "norm_diff_CV " + str(norm_diff_CV) + "\n"
    if ((norm_diff < tol_mu)and(norm_diff_CV < tol_CV)):
        merged_mu = mu_2 + (difference_mu/2)
        merdeg_CV = CV_2 + (difference_CV/2)
        return (merged_mu,merdeg_CV)
    else:
        return -1
    
if __name__ == "__main__":
    X, Y = create_10D_Data(100,0.9)
    X = X.T
    dimX, dimY = X.shape
    indexrandom = int(math.floor(np.random.uniform(0,dimY-1)))
    randomValue = X[:,indexrandom]
    XSorted = sortData(X,randomValue)
    X_S = XSorted[0]
    distances = XSorted[1]
    Probability = assignProbability(distances[:,0])
    
    min_dist_mu = distances[1,0]
    
    pl.figure()
    pl.plot(range(len(Probability)),Probability,'-b')
    indicess = findIndexMax(Probability,5)
    indicess.sort()
    for kk in indicess:
        pl.plot(kk,Probability[kk],'ro')
    pl.grid()
    
    list_mu = []    
#    pl.figure()
#    pl.plot(X_S[0,:],X_S[1,:],'bo',markeredgecolor = 'k',alpha = 0.3)

    for ll in range(len(indicess)):
        if (ll == 0):
#            pl.plot(X_S[0,int(indicess[ll]/2)],X_S[1,int(indicess[ll]/2)],'gs',markeredgecolor = 'k')
            list_mu.append(np.matrix([[X_S[0,int(indicess[ll]/2)]],[X_S[1,int(indicess[ll]/2)]]]))
        else:
            nindx = int(indicess[ll-1] + ((indicess[ll] - indicess[ll-1])/2))
#            pl.plot(X_S[0,nindx],X_S[1,nindx],'gs',markeredgecolor = 'k')
            list_mu.append([[X_S[0,nindx]],[X_S[1,nindx]]])
#    pl.plot(randomValue[0],randomValue[1],'rs')

    
#    res1 = prunning(mu1,mu2,CVx,CVy,min_dist_mu,0.04)
#    res2 = prunning(mu2,mu3,CVy,CVz,min_dist_mu,0.04)
#    media_X = np.sum(X,axis=1)/X.shape[1]
#    pl.plot(media_X[0],media_X[1],'rs')
#    CovarianceMatrix = np.matrix([[0.0, 0.0],[0.0, 0.0]])
#    
#    for ll in range(X.shape[1]):
#        ddiff = X[:,ll] - media_X
#        CovarianceMatrix = CovarianceMatrix + np.outer(ddiff,ddiff)
#    CovarianceMatrix = CovarianceMatrix/(190-1)
#    ReturnValue = growing(media_X,CovarianceMatrix,0.09)
#    
#    if(type(ReturnValue)==tuple):
#        listaNuevos_mu = ReturnValue[0]
#        listaNuevos_CV = ReturnValue[1]
