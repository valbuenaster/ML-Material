#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:32:34 2017

@author: ariel
"""

import matplotlib.pyplot as pl
import numpy as np
#from svmutil import *
import math

import homewok2_StoppingCriterionITERATIONS as hm2

if __name__ == "__main__":
    
    NClusters = 3
    NIterations = 40
    
    min_x = -6
    max_x = 6
    min_y = -6
    max_y = 6
    axesList = [min_x, max_x, min_y, max_y]

    mu1 = np.matrix([[0],[0]])
#    CV1 = np.matrix([[3, 1],[1, 2]])
    CV1 = np.matrix([[3, 0],[0, 1]])

    Data_1 = hm2.generateDistribution(mu1,CV1,200)
    D1Train, D1Test = hm2.splitTrainTest(Data_1,0.0)
    pl.figure()
    
    #Create a circle
    Angle = list(np.arange(0.0,2*math.pi,math.pi/40))
    Radius = 2.5
    Circle = []
    for angle in Angle:
        Circle.append(np.matrix([[Radius*math.cos(angle)],[Radius*math.sin(angle)]]))
    
    for aa in range(len(D1Train)):
        pl.plot(D1Train[aa][0],D1Train[aa][1],'o',color = '#ff0000',markeredgecolor = 'k',alpha = 0.35)
    for aa in range(len(Circle)):
        pl.plot(Circle[aa][0],Circle[aa][1],'s',color = '#0000ff',markeredgecolor = 'k',alpha = 0.35)
    pl.axis('square')
    pl.grid()
#    pl.savefig("Drawings/data.eps")
    pl.show()
    
    Mh_d = hm2.calculateMahalanobisdistance(Circle,mu1,CV1)
    
    pl.figure()
    pl.plot(Angle,Mh_d,'g')
    pl.grid()
    pl.show()