#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:40:59 2017

@author: Luis Ariel Valbuena Reyes
"""
#%%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
import matplotlib.colors as mplc
import numpy as np
#from svmutil import *
import math

def calculate_LogLikelihood(List_mu,List_CVMatrices,vectorPi,X):
    GMMResult_Data = GMM(List_mu,List_CVMatrices,vectorPi,X)
    summation = 0
    for element in GMMResult_Data:
        summation = summation + math.log(float(element))
        
    return summation

def calculate_probability_MultiGaussian(x,mu,CVm):
    size = len(x)
    detCV = abs(np.linalg.det(CVm))
    inverseCV = CVm.I
    Difference = np.matrix(x - mu)
    scaling = 1.0/( (math.pow(2*math.pi,size/2.0))*(math.sqrt(detCV)) )
    Argument = Difference.T*inverseCV*Difference
#    print(math.sqrt(np.dot(Difference.T,Difference)))
#    print(Argument)
    Result = scaling*math.pow(math.e,-0.5*Argument)
    return Result
    
def GMM(List_mu,List_CVMatrices,vectorPi,X):
    result = []
    for x in X:
        summation = 0
        for iter in range(len(List_CVMatrices)):
            mu = List_mu[iter]
            CVm = List_CVMatrices[iter]
            pi = vectorPi[iter]
            summation = summation + pi*calculate_probability_MultiGaussian(x,mu,CVm)
        result.append(summation)
    return result

def generateDistribution(muM,CVM,N):
    mu = []
    CV = CVM.tolist()
    for element in muM:
        mu.append(float(element[0]))
            
    x, y =  np.random.multivariate_normal(mu,CV,N).T
    return (x,y)

def calculate_mu(rk,r_ik,X):
    mu = 0
    for ii in range(len(X)):
        x = X[ii]
        rik = r_ik[ii]
        mu = mu + (rik*x)
    return mu/rk
    
def calculate_pk_rk(r_ik):
    N = len(r_ik)
    rk = 0
    for element in r_ik:
        rk = rk + element
    pk = rk/N
    return (rk,pk)
    
def calculate_CVM(rk,r_ik,X,muk):
    CVm = 0
    for ii in range(len(X)):
        x = X[ii]
        rik = r_ik[ii]
        CVm = CVm + (rik*np.outer(x,x))
    CVm = (CVm/rk)-np.outer(muk,muk)
    return CVm

def calculate_r_ik(vector_pi,X,mu,CV,k):
    r_ik = []
    for x in X:
        temp = vector_pi[k]*calculate_probability_MultiGaussian(x,mu[k],CV[k])
        resultGMM = GMM(mu,CV,vector_pi,[x])
        r_ik.append(temp/resultGMM[0])
    return r_ik
    
def updateParameters(vector_pi, X, muM,CVM, Features):
    r_k = []
    rk = []
    New_vector_pi = []
    New_muM = []
    New_CVM = []
    
    #E-step
    for kk in range(Features):
        r_k.append(calculate_r_ik(vector_pi,X,muM,CVM,kk))
        
    for kk in range(Features):
        dupleTemp = calculate_pk_rk(r_k[kk])
        rk = dupleTemp[0]
        New_vector_pi.append(dupleTemp[1])
        New_muM.append(calculate_mu(rk,r_k[kk],X))
        New_CVM.append(calculate_CVM(rk,r_k[kk],X,New_muM[-1]))
        
    return (New_vector_pi,New_muM,New_CVM)

def computeSurfaceZ(XX,YY,Lista_mu,Lista_CV,v_pi):
    dimX = len(XX)
    dimY = len(XX[0])
    Z = np.zeros((dimX,dimY))
    for ii in range(dimX):
        for jj in range(dimY):
            vect_x = np.matrix([[XX[ii][jj]],[YY[ii][jj]]])
            temp_GMM = GMM(Lista_mu,Lista_CV,v_pi,[vect_x])
            Z[ii][jj] = temp_GMM[0]
    return Z
            
#def updateParameters(vector_pi, X, muM,CVM, Features):
#    r_k = []
#    rk = []
#    New_vector_pi = []
#    New_muM = []
#    New_CVM = []
#    
#    for ii in range(Features):
#        r_i = []
#        for x in X:
#            mu = muM[ii]
#            CVm = CVM[ii]
#            temp = calculate_probability_MultiGaussian(x,mu,CVm)
#            resultGMM = GMM(muM,CVM,vector_pi,x)
#            r_i.append( (vector_pi[ii]*temp)/(resultGMM[0]))
#        r_k.append(r_i)
#
#        temp1 = 0
#        temp2 = 0
#        temp3 = 0
#        for jj in range(len(X)):
#            temp1 = temp1 + r_k[ii][jj]
#            x = X[jj]
#            temp2 = temp2 + r_k[ii][jj]*x
#            temp3 = temp3 + r_k[ii][jj]*np.outer(x,x)
#            
#        rk.append(temp1)
##        print('Value len(r_k[0])')
##        print(len(r_k[0]))
#        New_vector_pi.append(temp1/len(r_k[0]))
##        print('Value rk[-1]')
##        print(rk[-1])
#        New_muM.append(temp2/rk[-1])
#        New_CVM.append( (temp3/rk[-1]) - np.outer(New_muM[-1],New_muM[-1]) )
#        
#    return (New_vector_pi,New_muM,New_CVM)

       
if __name__ == "__main__":

    NClusters = 3

    mu1 = np.matrix([[1],[2]])
    mu2 = np.matrix([[-1],[-2]])
    mu3 = np.matrix([[3],[-3]])
    
    CV1 = np.matrix([[3, 1],[1, 2]])
    CV2 = np.matrix([[2, 0],[0, 1]])
    CV3 = np.matrix([[1, 0.3],[0.3, 1]])
    
#    x = np.matrix([[-0.5],[-0.8]])    
#    X = [x]
    #result = GMM(Lista_mu,Lista_CV,v_pi,X)#working...
    
    Data_1 = generateDistribution(mu1,CV1,100)
    Data_2 = generateDistribution(mu2,CV2,100)
    Data_3 = generateDistribution(mu3,CV3,200)
   
    min_x = -6
    max_x = 6
    min_y = -6
    max_y = 6
    axesList = [min_x, max_x, min_y, max_y]
    pl.clf()
    pl.plot(Data_1[0],Data_1[1],'bo')
    pl.plot(Data_2[0],Data_2[1],'ro')
    pl.plot(Data_3[0],Data_3[1],'go')
    pl.grid()
    pl.axis(axesList)
    pl.show()
    
    #data for the display
    Delta = 0.1
    xx = np.arange(min_x, max_x,Delta)
    yy = np.arange(min_y, max_y,Delta)
    XX, YY = np.meshgrid(xx,yy)
    #data for the display
    
    #Put all the generated points into a single list of vectors
    X = []
    for ii in range(len(Data_1[0])):
        X.append(np.matrix([[Data_1[0][ii]],[Data_1[1][ii]]]))
    
    for ii in range(len(Data_2[0])):
        X.append(np.matrix([[Data_2[0][ii]],[Data_2[1][ii]]]))
    
    for ii in range(len(Data_3[0])):
        X.append(np.matrix([[Data_3[0][ii]],[Data_3[1][ii]]]))
    
    mu1e = np.matrix([[-2],[-5]])   
    mu2e = np.matrix([[0.1],[4.5]]) 
    mu3e = np.matrix([[4.2],[0.2]]) 
    
#    indexrandom = int(math.floor(np.random.uniform(0,len(X)-1)))
#    mu1e = X[indexrandom] 
#    indexrandom = int(math.floor(np.random.uniform(0,len(X)-1)))  
#    mu2e = X[indexrandom] 
#    indexrandom = int(math.floor(np.random.uniform(0,len(X)-1)))
#    mu3e = X[indexrandom] 
    
    CV1e = np.matrix([[1, 0],[0, 1]])
    CV2e = np.matrix([[1, 0],[0, 1]])
    CV3e = np.matrix([[1, 0],[0, 1]])
        
    Lista_mu = [mu1e,mu2e,mu3e]
    Lista_CV = [CV1e,CV2e,CV3e]
    v_pi = [1/3.0, 1/3.0, 1/3.0]
    
#    #Lines to do pseudodebbuging
#    vector_pi = v_pi
#    muM = Lista_mu
#    CVM = Lista_CV
    path = [Lista_mu]
    ContourList = list(np.arange(0.001,0.02,0.001))
    
    ZZ = computeSurfaceZ(XX,YY,Lista_mu,Lista_CV,v_pi)
    
    pl.figure()
    mapaColores = pl.cm.gist_rainbow
#    CS = pl.contour(XX, YY, ZZ, ContourList, cmap = mapaColores)
#    pl.clabel(CS, inline=1, fontsize=10)
    pl.plot(Data_1[0],Data_1[1],'bo',markeredgecolor = 'k')
    pl.plot(Data_2[0],Data_2[1],'ro',markeredgecolor = 'k')
    pl.plot(Data_3[0],Data_3[1],'go',markeredgecolor = 'k')
    pl.axis(axesList)
    pl.grid()
    pl.show()
    
    Llike = calculate_LogLikelihood(Lista_mu,Lista_CV,v_pi,X)
    print "Initial Log likelihood "+ str(Llike) + "\n"
    
    ValuesLL = [Llike]
    MagnitudesDisplacement_mus =[]
#%%    
    for iteration in range(29):
        stringName = "Iteration: " + str(iteration) + "\n"
        print stringName
        Lista_mu_T_1 = list(Lista_mu)
        resultUpdate = updateParameters(v_pi, X,Lista_mu,Lista_CV,NClusters)
        v_pi = resultUpdate[0]
        Lista_mu = resultUpdate[1]
        path.append(Lista_mu)
        Lista_CV = []
        for CVm in resultUpdate[2]:
            Lista_CV.append(np.matrix(CVm))

        Llike = calculate_LogLikelihood(Lista_mu,Lista_CV,v_pi,X)
        print "Log likelihood "+ str(Llike) + "\n"
        ValuesLL.append(Llike)
        
        tempM = []
        for mu_n, mu_n1 in zip(Lista_mu,Lista_mu_T_1):
            tempM.append(math.sqrt(np.dot(mu_n.T - mu_n1.T,mu_n - mu_n1)))
        MagnitudesDisplacement_mus.append(tempM)
        
        #Drawing here
        ZZ = computeSurfaceZ(XX,YY,Lista_mu,Lista_CV,v_pi)
        pl.figure()
        CS = pl.contour(XX, YY, ZZ, ContourList, cmap = mapaColores)
        pl.clabel(CS, inline=1, fontsize=10)
#        pl.plot(Data_1[0],Data_1[1],'bo')
#        pl.plot(Data_2[0],Data_2[1],'ro')
#        pl.plot(Data_3[0],Data_3[1],'go')
        pl.axis(axesList)
        pl.grid()
        stringName = "Iteration_" + str(iteration)
        pl.savefig(stringName + '.eps')
        pl.show()
        
        figura = pl.figure()
        ax = figura.gca(projection = '3d')
        SS = ax.plot_surface(XX, YY, ZZ, linewidth=0.4, antialiased= True, cmap = pl.cm.plasma)
        figura.show()
#        figura.saverfig('Surface_'+stringName+'.eps')
        #Drawing here
        
    
    pl.clf()
    pl.plot(Data_1[0],Data_1[1],'bo',alpha = 0.35)
    pl.plot(Data_2[0],Data_2[1],'ro',alpha = 0.35)
    pl.plot(Data_3[0],Data_3[1],'go',alpha = 0.35)
    pl.grid()
    
    for element in path:
        pl.plot(element[0][0],element[0][1],'mo')
        pl.plot(element[1][0],element[1][1],'mp')
        pl.plot(element[2][0],element[2][1],'ms')
    pl.grid()
    pl.plot(mu1e[0],mu1e[1],'yo')
    pl.plot(mu2e[0],mu2e[1],'yp')
    pl.plot(mu3e[0],mu3e[1],'ys')
    pl.axis(axesList)
    
    for ii in range(1,len(path)):
        pl.quiver(float(path[ii-1][0][0]),float(path[ii-1][0][1]),float(path[ii][0][0]-path[ii-1][0][0]),float(path[ii][0][1]-path[ii-1][0][1]),angles='xy', scale_units='xy', scale=1)
        pl.quiver(path[ii-1][1][0],path[ii-1][1][1],path[ii][1][0]-path[ii-1][1][0],path[ii][1][1]-path[ii-1][1][1],angles='xy', scale_units='xy', scale=1)
        pl.quiver(path[ii-1][2][0],path[ii-1][2][1],path[ii][2][0]-path[ii-1][2][0],path[ii][2][1]-path[ii-1][2][1],angles='xy', scale_units='xy', scale=1)
    pl.grid()
    pl.savefig('ui_Paths.eps')
#    pl.savefig('Preliminary.png')
    pl.show()
    
    pl.figure()
    pl.plot(range(len(ValuesLL)),ValuesLL,'-bs')
#    pl.plot(range(len(ValuesLL)),[(-1*elem) for elem in ValuesLL])
#    pl.yscale('log')
    pl.grid()
    pl.show()
    
    mu1_disp = []
    mu2_disp = []
    mu3_disp = []
    
    for ll in MagnitudesDisplacement_mus:
        mu1_disp.append(ll[0])
        mu2_disp.append(ll[1])
        mu3_disp.append(ll[2])
        
    pl.figure()
    uu1 = pl.plot(range(len(mu1_disp)),mu1_disp,'-r*',label = 'mu_1')
    uu2 = pl.plot(range(len(mu2_disp)),mu2_disp,'-bo',label = 'mu_2')
    uu3 = pl.plot(range(len(mu3_disp)),mu3_disp,'-gs',label = 'mu_3')
    pl.yscale('log')
    pl.grid()
#    pl.legend(handles = [uu1, uu2, uu3])
    pl.legend()
    pl.title('Displacements of means')
    pl.savefig('EvolutionMuDisplacement.eps')
    pl.show()
    
    Xscatter1 = []
    Yscatter1 = []
    Zscatter1 = []
    for tx,ty in zip(Data_1[0],Data_1[1]):
        Xscatter1.append(tx)
        Yscatter1.append(ty)
        vect_temp = np.matrix([[tx],[ty]])
        temp_GMM = GMM(Lista_mu,Lista_CV,v_pi,[vect_temp])
        Zscatter1.append(temp_GMM[0])
     
    Xscatter2 = []
    Yscatter2 = []
    Zscatter2 = []
    for tx,ty in zip(Data_2[0],Data_2[1]):
        Xscatter2.append(tx)
        Yscatter2.append(ty)
        vect_temp = np.matrix([[tx],[ty]])
        temp_GMM = GMM(Lista_mu,Lista_CV,v_pi,[vect_temp])
        Zscatter2.append(temp_GMM[0])
    
    Xscatter3 = []
    Yscatter3 = []
    Zscatter3 = []
    for tx,ty in zip(Data_3[0],Data_3[1]):
        Xscatter3.append(tx)
        Yscatter3.append(ty)
        vect_temp = np.matrix([[tx],[ty]])
        temp_GMM = GMM(Lista_mu,Lista_CV,v_pi,[vect_temp])
        Zscatter3.append(temp_GMM[0])
    
#    pl.figure()
#    pl.scatter(Xscatter1,Yscatter1,s=1500*Zscatter1, c = 'b')
#    pl.scatter(Xscatter2,Yscatter2,s=1500*Zscatter2, c = 'r')
#    pl.scatter(Xscatter3,Yscatter3,s=1500*Zscatter3, c = 'g')
#    pl.axis(axesList)
#    pl.grid()
##    pl.savefig('DataScatterPlot.eps')
#    pl.savefig('DataScatterPlot.png')
#    pl.show()
    