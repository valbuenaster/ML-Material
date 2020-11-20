#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:17:22 2017

@author: ariel
"""

import matplotlib.pyplot as pl
import numpy as np
from svmutil import *

if __name__ == "__main__":
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
    pl.savefig('SimulatedProccess.eps')
    pl.show()
    
    yTrain = []
    obsTrain = []
    signalTrain = []
    yTest = []
    obsTest = []
    signalTest = []
    
    for ii in range(len(x)):
        if(np.random.uniform(0,1,1)[0]>0.18):
            yTrain.append(x[ii])
            obsTrain.append(observed[ii])
            signalTrain.append(y[ii])
        else:
            yTest.append(x[ii])
            obsTest.append(observed[ii])
            signalTest.append(y[ii])
    
    prob = svm_problem(obsTrain,yTrain)#prob = svm_problem(y,x,isKernel=True)%
    param = svm_parameter('-t 0 -s 3 -p 0.0 -c 40')
    model = svm_train(prob,param)
    
    #use the train data
    p_label, p_acc, p_val = svm_predict(obsTrain,yTrain, model, '-b 0')
    
    pl.figure()
    pl.clf()
    pl.plot(range(len(p_label)),p_label,'-r')
    pl.plot(range(len(obsTrain)),obsTrain,'-b')
    pl.plot(range(len(signalTrain)),signalTrain,'-g')
    pl.title('Train data')
    pl.grid()
    pl.savefig('TrainingData.eps')
    pl.show()
    
    errorTrainData = []
    for ii in range(len(p_label)):
        errorTrainData.append(abs(p_label[ii]-signalTrain[ii][0]))
        
    pl.figure()
    pl.clf()
    pl.plot(range(len(p_label)),errorTrainData,'-r')
    pl.title('Error with training data.')
    pl.grid()
    pl.savefig('ErrorTrainingSamples.eps')
    pl.show()
    
    p_labelTest, p_accTest, p_valTest = svm_predict(obsTest,yTest, model, '-b 0')
    
    pl.figure()
    pl.clf()
    pl.plot(range(len(p_labelTest)),p_labelTest,'-r')
    pl.plot(range(len(obsTest)),obsTest,'-b')
    pl.plot(range(len(signalTest)),signalTest,'-g')
    pl.title('Test data.')
    pl.grid()
    pl.savefig('TestData.eps')
    pl.show()
    
    errorTestData = []
    for ii in range(len(p_labelTest)):
        errorTestData.append(abs(p_labelTest[ii]-signalTest[ii][0]))
        
    pl.figure()
    pl.clf()
    pl.plot(range(len(p_labelTest)),errorTestData,'-r')
    pl.title('Error with testing data.')
    pl.grid()
    pl.savefig('ErrorTestSamples.eps')
    pl.show()