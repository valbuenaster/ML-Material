#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:26:29 2017

@author: Luis Ariel Valbuena Reyes
"""

import matplotlib.pyplot as pl
import numpy as np
from svmutil import *
from math import sqrt
from math import isnan

def errorVector(List1,List2):
    Error = []
    for ii in range(len(List1)):
        if((isnan(List1[ii])==False)&(isnan(List2[ii])==False)):
            Error.append(List1[ii]-List2[ii])
    return Error

def meanSquareError(errorList):
    mse = 0.0
    for ii in range(len(errorList)):
        mse += errorList[ii]*errorList[ii]
    return mse/len(errorList)

def returnPrecomputedKernel(Ztransp,gamma):
    M = len(Ztransp[0])
    N = len(Ztransp)
    
    K = (gamma*(np.identity(N))) + np.dot(Ztransp,Ztransp.T)
    #K =  np.dot(Ztransp,Ztransp.T)
    return K
    
def formatKernelforSVMLibrary(K):
    L = len(K)
    return np.column_stack((np.array(range(1,L+1)),K))

def reshapeData(X,Y,p,q):
    lenght = len(X)
    Matrix = []
    
    for ii in range(p,lenght):
        temp = []
        for jj in range(ii-p,ii+1):
            temp.append(X[jj][0])
        for kk in range(ii-1-q,ii):
            temp.append(Y[kk][0])
            
        Matrix.append(temp)
    return Matrix
    
def reformatLabels(obsTrain,indexObs):
    L = len(obsTrain)
    return obsTrain[indexObs+1:L]
    
def convert2normalList(arrayNP):
    listt = []
    for element in arrayNP:
        listt.append(element[0])
    return listt
    
if __name__ == "__main__":

    gamma = 0.1
    xtemp = np.random.normal(0,1,100)
    xtempL = list(xtemp)
    
    x =[]
    x.append([0])
    x.append([0])
    
    for value in xtempL:
        x.append([value])
        
    nvalues = range(2,len(x))    
    y = []
    y.append([0])
    y.append([0])
    observed = []

    for n in nvalues:
        y.append([(3*x[n][0]) - (0.5*x[n-1][0]) + (0.5*x[n-2][0]) + (0.03*y[n-1][0]) - (0.01*y[n-2][0]) ])
        observed.append(y[n] + np.random.normal(0,0.1,1)[0])
        
    x.pop(0)
    x.pop(0)
    y.pop(0)
    y.pop(0)
    
    pl.clf()
    pl.plot(range(0,100),x,'b-')
    pl.plot(range(0,100),y,'r-')
    pl.plot(range(0,100),observed,'g-')
    pl.grid()
    pl.title('Simulated Proccesss')
    pl.savefig('SimulatedProccess.eps')
    pl.show()
    
    yTrain = y
    obsTrain = observed
    xTrain = x
    
#    yTrain = []
#    obsTrain = []
#    xTrain = []
#    yTest = []
#    obsTest = []
#    xTest = []
#    for ii in range(len(x)):
#        if(ii<80):
#            xTrain.append(x[ii])
#            obsTrain.append(observed[ii])
#            yTrain.append(y[ii])
#        else:
#            xTest.append(x[ii])
#            obsTest.append(observed[ii])
#            yTest.append(y[ii])

        
#    xTrain.append(x[0:80])
#    obsTrain.append(observed[0:80])
#    yTrain.append(y[0:80]) 
#    
#    xTest.append(x[80:100])
#    obsTest.append(observed[80:100])
#    yTest.append(y[80:100])      
    
    indexObs = 1
    indexX = 2
    z = reshapeData(xTrain,obsTrain,indexX,indexObs)#I think we need to use the observed signal
    Zt_Training = np.array(z) #Transposed matrix
    Kernel = returnPrecomputedKernel(Zt_Training,gamma)
    indexedKernelTraining = formatKernelforSVMLibrary(Kernel)
    OBSTraining = reformatLabels(obsTrain,indexObs)
    tempYTraining = reformatLabels(yTrain,indexObs)
    
    YTraining = []
    for element in tempYTraining:
        YTraining.append(element[0])
    
#    z = reshapeData(xTest,obsTest,indexX,indexObs)#I think we need to use the observed signal
#    Zt_Test = np.array(z) #Transposed matrix
#    Kernel = returnPrecomputedKernel(Zt_Test,gamma)
#    indexedKernelTest = formatKernelforSVMLibrary(Kernel)
#    OBSTest = reformatLabels(obsTest,indexObs)
    
    valuesC = range(1,101,1)
    
    rmsTraining = []
    rmsTesting = []
    
    OBSlistTraining = convert2normalList(OBSTraining)
    
#    OBSlistTest = convert2normalList(OBSTest)
    
    for C in valuesC:
        argumentString = '-t 4 -s 3 -p 0.001 -c '+ str(C)
        model = svm_train(OBSTraining, [list(row) for row in indexedKernelTraining], argumentString)
        
        p_label, p_acc, p_val = svm_predict(OBSTraining,[list(row) for row in indexedKernelTraining], model)
        error = errorVector(OBSlistTraining,p_label)
#        error = errorVector(YTraining,p_label)
        rmsTraining.append(meanSquareError(error))
        
#        p_label, p_acc, p_val = svm_predict(OBSTest,[list(row) for row in indexedKernelTest], model)
#        error = errorVector(OBSlistTest,p_label)
#        rmsTesting.append(meanSquareError(error))
        
    pl.figure()
    pl.clf()
    pl.plot(valuesC,rmsTraining,'-r')
#    pl.plot(valuesC,rmsTesting,'-b')
#    pl.plot(range(len(signalTest)),signalTest,'-g')
    pl.yscale('log')
    pl.title('rms error.')
    pl.grid()
    pl.savefig('rms error.eps')
    pl.show()   