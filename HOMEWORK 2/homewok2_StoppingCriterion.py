#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:02:33 2017

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

def updateParameters_K_MEANS(X,muM,CVM,Features): 
    New_muM = []
    New_CVM = []   
    r_ik = calculate_r_ik_K_MEANS(X,muM,CVM,Features)
    
    for kk in range(Features):
        dupleTemp = calculate_pk_rk(r_ik[kk])
        rk = dupleTemp[0]
        New_muM.append(calculate_mu(rk,r_ik[kk],X))
        New_CVM.append(calculate_CVM(rk,r_ik[kk],X,New_muM[-1]))
        
    return (New_muM,New_CVM)
        
    
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

def calculateMahalanobisdistance(X,mu,CV):
    arrayDistances = []
    inverseCV = CV.I
    for x in X:
        difference = mu - x
        arrayDistances.append(math.sqrt(difference.T*inverseCV*difference))
    return arrayDistances
        
def calculate_r_ik_K_MEANS(X,mu,CV,Features):
    r_ik = []
    for x in X:
        distances = []
        rr_ik = []
        for ii in range(Features):
            distances.append(calculateMahalanobisdistance([x],mu[ii],CV[ii]))
        leastNorm = min(distances)
        for dist in distances:
            if (dist == leastNorm):
                rr_ik.append(1)
            else:
                rr_ik.append(0)

        r_ik.append(rr_ik)
    return r_ik

def splitTrainTest(Data,threshold):
    D_Train = []
    D_Test = []
    for ii in range(len(Data[0])):
        if(np.random.uniform()>threshold):
            D_Train.append(np.matrix([[Data[0][ii]],[Data[1][ii]]]))
        else:
            D_Test.append(np.matrix([[Data[0][ii]],[Data[1][ii]]]))
    return (D_Train,D_Test)

                
if __name__ == "__main__":

    NClusters = 3
    NIterations = 40

    mu1 = np.matrix([[1],[2]])
    mu2 = np.matrix([[-1],[-2]])
    mu3 = np.matrix([[3],[-3]])
    
    CV1 = np.matrix([[3, 1],[1, 2]])
    CV2 = np.matrix([[2, 0],[0, 1]])
    CV3 = np.matrix([[1, 0.3],[0.3, 1]])

    
    Data_1 = generateDistribution(mu1,CV1,100)
    Data_2 = generateDistribution(mu2,CV2,100)
    Data_3 = generateDistribution(mu3,CV3,200)
    
    threshold = 0.2
    min_x = -6
    max_x = 6
    min_y = -6
    max_y = 6
    axesList = [min_x, max_x, min_y, max_y]
    
    #data for the display
    Delta = 0.1
    xx = np.arange(min_x, max_x,Delta)
    yy = np.arange(min_y, max_y,Delta)
    XX, YY = np.meshgrid(xx,yy)
    #data for the display
    
    
    D1Train, D1Test = splitTrainTest(Data_1,threshold)
    D2Train, D2Test = splitTrainTest(Data_2,threshold)
    D3Train, D3Test = splitTrainTest(Data_3,threshold)
   
    #Put all the generated points into a single list of vectors        
        
    XTrain = D1Train + D2Train + D3Train
    XTest = D1Test + D2Test + D3Test

    
    mu1e = np.matrix([[-2],[-5]])   
    mu2e = np.matrix([[0.1],[4.5]]) 
    mu3e = np.matrix([[4.2],[0.2]]) 
    
    CV1e = np.matrix([[1, 0],[0, 1]])
    CV2e = np.matrix([[1, 0],[0, 1]])
    CV3e = np.matrix([[1, 0],[0, 1]])
        
    Lista_mu = [mu1e,mu2e,mu3e]
    Lista_CV = [CV1e,CV2e,CV3e]
    v_pi = [1/3.0, 1/3.0, 1/3.0]
    

    pathGMM = [Lista_mu]
    ContourList = list(np.arange(0.001,0.02,0.001))
    
    ZZ = computeSurfaceZ(XX,YY,Lista_mu,Lista_CV,v_pi)
    
    pl.figure()
    mapaColores = pl.cm.gist_rainbow
#    CS = pl.contour(XX, YY, ZZ, ContourList, cmap = mapaColores)
#    pl.clabel(CS, inline=1, fontsize=10)
    for aa in range(len(D1Train)):
        pl.plot(D1Train[aa][0],D1Train[aa][1],'o',color = '#ff0000',markeredgecolor = 'k',alpha = 0.35)#r
    for aa in range(len(D2Train)):
        pl.plot(D2Train[aa][0],D2Train[aa][1],'o',color = '#00ff00',markeredgecolor = 'k',alpha = 0.35)#g
    for aa in range(len(D3Train)):
        pl.plot(D3Train[aa][0],D3Train[aa][1],'o',color = '#0000ff',markeredgecolor = 'k',alpha = 0.35)#b
    
    for aa in range(len(D1Test)):
        pl.plot(D1Test[aa][0],D1Test[aa][1],'o',color = '#9b0000',markeredgecolor = 'k')#r
    for aa in range(len(D2Test)):
        pl.plot(D2Test[aa][0],D2Test[aa][1],'o',color = '#009b00',markeredgecolor = 'k')#g
    for aa in range(len(D3Test)):
        pl.plot(D3Test[aa][0],D3Test[aa][1],'o',color = '#00009b',markeredgecolor = 'k')#b
    
    pl.axis(axesList)
    pl.grid()
#    pl.savefig("Drawings/data.eps")
    pl.show()
    
    Llike = calculate_LogLikelihood(Lista_mu,Lista_CV,v_pi,XTest)
    print "Initial Log likelihood "+ str(Llike) + "\n"
    
    ValuesLLGMMTrain = [calculate_LogLikelihood(Lista_mu,Lista_CV,v_pi,XTrain)]
    ValuesLLGMMTest = [calculate_LogLikelihood(Lista_mu,Lista_CV,v_pi,XTest)]
    MagnitudesDisplacement_musGMM =[]
#%% 
    iteration = 1
#    while(True):   
    while(iteration < NIterations):
        stringName = "Iteration: " + str(iteration) + "\n"
        
        print stringName
        Lista_mu_T_1 = list(Lista_mu)
        resultUpdate = updateParameters(v_pi, XTrain,Lista_mu,Lista_CV,NClusters)
        v_pi = resultUpdate[0]
        Lista_mu = resultUpdate[1]
        pathGMM.append(Lista_mu)
        Lista_CV = []
        for CVm in resultUpdate[2]:
            Lista_CV.append(np.matrix(CVm))

        ValuesLLGMMTrain.append(calculate_LogLikelihood(Lista_mu,Lista_CV,v_pi,XTrain))
        ValuesLLGMMTest.append(calculate_LogLikelihood(Lista_mu,Lista_CV,v_pi,XTest))
        print "Log likelihood "+ str(ValuesLLGMMTest[-1]) + "\n"
        
        tempM = []
        for mu_n, mu_n1 in zip(Lista_mu,Lista_mu_T_1):
            tempM.append(math.sqrt(np.dot(mu_n.T - mu_n1.T,mu_n - mu_n1)))
        MagnitudesDisplacement_musGMM.append(tempM)
        NormMagnitudeDisplacement = math.sqrt( pow(MagnitudesDisplacement_musGMM[-1][0],2) + pow(MagnitudesDisplacement_musGMM[-1][1],2) + pow(MagnitudesDisplacement_musGMM[-1][2],2) )
        print "norm(tempM)\n"
        print NormMagnitudeDisplacement
        
        #Drawing here
        ZZ = computeSurfaceZ(XX,YY,Lista_mu,Lista_CV,v_pi)
        pl.figure()
        CS = pl.contour(XX, YY, ZZ, ContourList, cmap = mapaColores)
        pl.clabel(CS, inline=1, fontsize=10)

        pl.axis(axesList)
        pl.grid()
        stringName = "GMM Iteration_" + str(iteration)
        pl.title(stringName)
#        pl.savefig('Drawings/'+stringName + '.eps')
        iteration = iteration + 1
        pl.show()
        
        figura = pl.figure()
        ax = figura.gca(projection = '3d')
        SS = ax.plot_surface(XX, YY, ZZ, linewidth=0.4, antialiased= True, cmap = pl.cm.plasma)
        cset = ax.contour(XX, YY, ZZ, zdir='x', offset=min_x, cmap=pl.cm.coolwarm)
        cset = ax.contour(XX, YY, ZZ, zdir='y', offset=max_y, cmap=pl.cm.coolwarm)

        ax.set_xlabel('X')
        ax.set_xlim(min_x, max_x)
        ax.set_ylabel('Y')
        ax.set_ylim(min_y, max_y)
        ax.set_zlabel('GMM')
#        ax.set_zlim(-0.46, 0.35)
        figura.show()
#        figura.saverfig('Surface_'+stringName+'.eps')
        #Drawing here
#        if(NormMagnitudeDisplacement < 0.01):
#            break
        
    
    pl.figure()
    pl.plot(Data_1[0],Data_1[1],'bo',alpha = 0.35,markeredgecolor = 'k')
    pl.plot(Data_2[0],Data_2[1],'ro',alpha = 0.35,markeredgecolor = 'k')
    pl.plot(Data_3[0],Data_3[1],'go',alpha = 0.35,markeredgecolor = 'k')
    pl.grid()
    
    for element in pathGMM:
        pl.plot(element[0][0],element[0][1],'mo',markeredgecolor = 'k')
        pl.plot(element[1][0],element[1][1],'mp',markeredgecolor = 'k')
        pl.plot(element[2][0],element[2][1],'ms',markeredgecolor = 'k')
    pl.grid()
    pl.plot(mu1e[0],mu1e[1],'yo',markeredgecolor = 'k')
    pl.plot(mu2e[0],mu2e[1],'yp',markeredgecolor = 'k')
    pl.plot(mu3e[0],mu3e[1],'ys',markeredgecolor = 'k')
    pl.axis(axesList)
    
    for ii in range(1,len(pathGMM)):
        pl.quiver(pathGMM[ii-1][0][0],pathGMM[ii-1][0][1],pathGMM[ii][0][0]-pathGMM[ii-1][0][0],pathGMM[ii][0][1]-pathGMM[ii-1][0][1],angles='xy', scale_units='xy', scale=1)
        pl.quiver(pathGMM[ii-1][1][0],pathGMM[ii-1][1][1],pathGMM[ii][1][0]-pathGMM[ii-1][1][0],pathGMM[ii][1][1]-pathGMM[ii-1][1][1],angles='xy', scale_units='xy', scale=1)
        pl.quiver(pathGMM[ii-1][2][0],pathGMM[ii-1][2][1],pathGMM[ii][2][0]-pathGMM[ii-1][2][0],pathGMM[ii][2][1]-pathGMM[ii-1][2][1],angles='xy', scale_units='xy', scale=1)
    pl.grid()
    pl.savefig('Drawings/'+'ui_Paths_GMM.eps')

    pl.show()
    
    pl.figure()
#    pl.plot(range(1,len(ValuesLLGMMTrain)),ValuesLLGMMTrain[1:len(ValuesLLGMMTrain)],'-bs',markeredgecolor = 'k')
    pl.plot(range(1,len(ValuesLLGMMTest)),ValuesLLGMMTest[1:len(ValuesLLGMMTest)],'-rs',markeredgecolor = 'k')
#    pl.plot(range(1,len(ValuesLLGMMTrain)),[-elem for elem in ValuesLLGMMTrain[1:len(ValuesLLGMMTrain)]],'-bs',markeredgecolor = 'k')
#    pl.plot(range(1,len(ValuesLLGMMTest)),[-elem for elem in ValuesLLGMMTest[1:len(ValuesLLGMMTest)]],'-rs',markeredgecolor = 'k')
#    pl.yscale('log')
    pl.grid()
    pl.title('log likelihood GMM')
    pl.savefig('Drawings/'+'loglikelihookGMM.eps')
    pl.show()
    
    mu1_disp = []
    mu2_disp = []
    mu3_disp = []
    norm_disp = []
    
    for ll in MagnitudesDisplacement_musGMM:
        mu1_disp.append(ll[0])
        mu2_disp.append(ll[1])
        mu3_disp.append(ll[2])
        norm_disp.append(math.sqrt( pow(ll[0],2) + pow(ll[1],2) + pow(ll[2],2) ))
        
    pl.figure()
    uu1 = pl.plot(range(len(mu1_disp)),mu1_disp,'-r*',markeredgecolor = 'k',label = 'mu_1')
    uu2 = pl.plot(range(len(mu2_disp)),mu2_disp,'-bo',markeredgecolor = 'k',label = 'mu_2')
    uu3 = pl.plot(range(len(mu3_disp)),mu3_disp,'-gs',markeredgecolor = 'k',label = 'mu_3')
    unorm = pl.plot(range(len(norm_disp)),norm_disp,'-m^',markeredgecolor = 'k',label = 'Norm')
    pl.yscale('log')
    pl.grid()
#    pl.legend(handles = [uu1, uu2, uu3])
    pl.legend()
    pl.title('Displacements of means')
    pl.savefig('Drawings/'+'EvolutionMuDisplacement_GMM.eps')
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
        
#NEED TO EXPORT THIS DATA TO MATLAB TO DRAW THE SCATTER PLOT
    
#    pl.figure()
#    pl.scatter(Xscatter1,Yscatter1,s=1500*Zscatter1, c = 'b')
#    pl.scatter(Xscatter2,Yscatter2,s=1500*Zscatter2, c = 'r')
#    pl.scatter(Xscatter3,Yscatter3,s=1500*Zscatter3, c = 'g')
#    pl.axis(axesList)
#    pl.grid()
##    pl.savefig('DataScatterPlot.eps')
#    pl.savefig('DataScatterPlot.png')
#    pl.show()
    
#%%
#    LevelsClassification = [0.011, 0.012, 0.013, 0.014, 0.015]
    LevelsClassification = [0.009, 0.011]
    pl.figure()
    CS = pl.contour(XX, YY, ZZ, LevelsClassification, cmap = mapaColores)
    pl.plot(Data_1[0],Data_1[1],'bo',markeredgecolor = 'k')
    pl.plot(Data_2[0],Data_2[1],'ro',markeredgecolor = 'k')
    pl.plot(Data_3[0],Data_3[1],'go',markeredgecolor = 'k')
    
    pl.clabel(CS, inline=1, fontsize=10)
    pl.axis(axesList)
    pl.grid()
    pl.savefig('Drawings/'+'LevelsClassification.eps')
    pl.show()
    
#%% 
#    #Drawing first probability gaussian
    SelectZ = 0    
    ZZg1 = computeSurfaceZ(XX,YY,[Lista_mu[SelectZ]],[Lista_CV[SelectZ]],[v_pi[SelectZ]])
#    pl.figure()
#    CS = pl.contour(XX, YY, ZZg1, ContourList, cmap = mapaColores)
#    pl.clabel(CS, inline=1, fontsize=10)
#    pl.axis(axesList)
#    pl.grid()
#    
#
#    #Drawing second probability gaussian
    SelectZ = 1    
    ZZg2 = computeSurfaceZ(XX,YY,[Lista_mu[SelectZ]],[Lista_CV[SelectZ]],[v_pi[SelectZ]])
#    pl.figure()
#    CS = pl.contour(XX, YY, ZZg2, ContourList, cmap = mapaColores)
#    pl.clabel(CS, inline=1, fontsize=10)
#    pl.axis(axesList)
#    pl.grid()
#    
#
#    #Drawing thrid probability gaussian
    SelectZ = 2    
    ZZg3 = computeSurfaceZ(XX,YY,[Lista_mu[SelectZ]],[Lista_CV[SelectZ]],[v_pi[SelectZ]])
#    pl.figure()
#    CS = pl.contour(XX, YY, ZZg3, ContourList, cmap = mapaColores)
#    pl.clabel(CS, inline=1, fontsize=10)
#    pl.axis(axesList)
#    pl.grid()
#%%  
    #All individual probability gaussians
    pl.figure()
    pl.plot(Data_1[0],Data_1[1],'bo',markeredgecolor = 'k')
    pl.plot(Data_2[0],Data_2[1],'ro',markeredgecolor = 'k')
    pl.plot(Data_3[0],Data_3[1],'go',markeredgecolor = 'k')
    
    CS = pl.contour(XX, YY, ZZg1, [0.0085], cmap = mapaColores)#Red
    pl.clabel(CS, inline=1, fontsize=10)
    CS = pl.contour(XX, YY, ZZg2, [0.002], cmap = mapaColores)#Blue
    pl.clabel(CS, inline=1, fontsize=10)
    CS = pl.contour(XX, YY, ZZg3, [0.006], cmap = mapaColores)#Green
    pl.clabel(CS, inline=1, fontsize=10)
    pl.axis(axesList)
    pl.grid()
    pl.savefig('Drawings/'+'LevelsClassificationInsight.eps')

#%% Using k-means
#    indexrandom = int(math.floor(np.random.uniform(0,len(X)-1)))
#    mu1e = X[indexrandom] 
#    indexrandom = int(math.floor(np.random.uniform(0,len(X)-1)))  
#    mu2e = X[indexrandom] 
#    indexrandom = int(math.floor(np.random.uniform(0,len(X)-1)))
#    mu3e = X[indexrandom] 
#    
#    CV1e = np.matrix([[1, 0],[0, 1]])
#    CV2e = np.matrix([[1, 0],[0, 1]])
#    CV3e = np.matrix([[1, 0],[0, 1]])
    
    Lista_mu = [mu1e,mu2e,mu3e]
    Lista_CV = [CV1e,CV2e,CV3e]
    pathKmeans = [Lista_mu]
    iteration = 1
    MagnitudesDisplacement_musKmeans =[]
    ValuesLLKmeansTrain = [calculate_LogLikelihood(Lista_mu,Lista_CV,v_pi,XTrain)]
    ValuesLLKmeansTest = [calculate_LogLikelihood(Lista_mu,Lista_CV,v_pi,XTest)]
    MagnitudesDisplacement_musKmeans =[]    
#    while(True):
    while(iteration < NIterations):
        stringName = "Iteration: " + str(iteration) + "\n"
        print stringName
        Lista_mu_T_1 = list(Lista_mu)
        resultUpdate = updateParameters(v_pi, XTrain,Lista_mu,Lista_CV,NClusters)
        v_pi = resultUpdate[0]
        Lista_mu = resultUpdate[1]
        pathKmeans.append(Lista_mu)
        Lista_CV = []
        for CVm in resultUpdate[2]:
            Lista_CV.append(np.matrix(CVm))
            
        ValuesLLKmeansTrain.append(calculate_LogLikelihood(Lista_mu,Lista_CV,v_pi,XTrain))
        ValuesLLKmeansTest.append(calculate_LogLikelihood(Lista_mu,Lista_CV,v_pi,XTest))
        print "Log likelihood "+ str(ValuesLLGMMTest[-1]) + "\n"
        
        tempM = []
        for mu_n, mu_n1 in zip(Lista_mu,Lista_mu_T_1):
            tempM.append(math.sqrt(np.dot(mu_n.T - mu_n1.T,mu_n - mu_n1)))
        MagnitudesDisplacement_musKmeans.append(tempM)
        NormMagnitudeDisplacement = math.sqrt( pow(MagnitudesDisplacement_musKmeans[-1][0],2) + pow(MagnitudesDisplacement_musKmeans[-1][1],2) + pow(MagnitudesDisplacement_musKmeans[-1][2],2) )
        print "norm(tempM)\n"
        print NormMagnitudeDisplacement
        
        #Drawing here
        ZZ = computeSurfaceZ(XX,YY,Lista_mu,Lista_CV,v_pi)
        pl.figure()
        CS = pl.contour(XX, YY, ZZ, ContourList, cmap = mapaColores)
        pl.clabel(CS, inline=1, fontsize=10)

        pl.axis(axesList)
        pl.grid()
        stringName = "K-Means Iteration_" + str(iteration)
        pl.title(stringName)
        pl.savefig('Drawings/'+stringName + '.eps')
        pl.show()
        
        iteration = iteration + 1
        
        figura = pl.figure()
        ax = figura.gca(projection = '3d')
        SS = ax.plot_surface(XX, YY, ZZ, linewidth=0.4, antialiased= True, cmap = pl.cm.plasma)
        cset = ax.contour(XX, YY, ZZ, zdir='x', offset=min_x, cmap=pl.cm.coolwarm)
        cset = ax.contour(XX, YY, ZZ, zdir='y', offset=max_y, cmap=pl.cm.coolwarm)

        ax.set_xlabel('X')
        ax.set_xlim(min_x, max_x)
        ax.set_ylabel('Y')
        ax.set_ylim(min_y, max_y)
        ax.set_zlabel('K-Means')
#        ax.set_zlim(-0.46, 0.35)
        figura.show()
#        figura.saverfig('Surface_'+stringName+'.eps')
        #Drawing here
#        if(NormMagnitudeDisplacement < 0.01):
#            break
#%%        
    pl.figure()
    pl.plot(Data_1[0],Data_1[1],'bo',alpha = 0.35,markeredgecolor = 'k')
    pl.plot(Data_2[0],Data_2[1],'ro',alpha = 0.35,markeredgecolor = 'k')
    pl.plot(Data_3[0],Data_3[1],'go',alpha = 0.35,markeredgecolor = 'k')
    pl.grid()
    
    for element in pathKmeans:
        pl.plot(element[0][0],element[0][1],'mo',markeredgecolor = 'k')
        pl.plot(element[1][0],element[1][1],'mp',markeredgecolor = 'k')
        pl.plot(element[2][0],element[2][1],'ms',markeredgecolor = 'k')
    pl.grid()
    pl.plot(mu1e[0],mu1e[1],'yo',markeredgecolor = 'k')
    pl.plot(mu2e[0],mu2e[1],'yp',markeredgecolor = 'k')
    pl.plot(mu3e[0],mu3e[1],'ys',markeredgecolor = 'k')
    pl.axis(axesList)
    
    for ii in range(1,len(pathKmeans)):
        pl.quiver(pathKmeans[ii-1][0][0],pathKmeans[ii-1][0][1],pathKmeans[ii][0][0]-pathKmeans[ii-1][0][0],pathKmeans[ii][0][1]-pathKmeans[ii-1][0][1],angles='xy', scale_units='xy', scale=1)
        pl.quiver(pathKmeans[ii-1][1][0],pathKmeans[ii-1][1][1],pathKmeans[ii][1][0]-pathKmeans[ii-1][1][0],pathKmeans[ii][1][1]-pathKmeans[ii-1][1][1],angles='xy', scale_units='xy', scale=1)
        pl.quiver(pathKmeans[ii-1][2][0],pathKmeans[ii-1][2][1],pathKmeans[ii][2][0]-pathKmeans[ii-1][2][0],pathKmeans[ii][2][1]-pathKmeans[ii-1][2][1],angles='xy', scale_units='xy', scale=1)
    pl.grid()
    pl.savefig('Drawings/'+'ui_Paths_K_MEANS.eps')
#    pl.savefig('Preliminary.png')
    pl.show()
    
    pl.figure()
#    pl.plot(range(1,len(ValuesLLGMMTrain)),ValuesLLGMMTrain[1:len(ValuesLLGMMTrain)],'-bs',markeredgecolor = 'k')
    pl.plot(range(1,len(ValuesLLKmeansTest)),ValuesLLKmeansTest[1:len(ValuesLLKmeansTest)],'-rs',markeredgecolor = 'k')
#    pl.plot(range(1,len(ValuesLLGMMTrain)),[-elem for elem in ValuesLLGMMTrain[1:len(ValuesLLGMMTrain)]],'-bs',markeredgecolor = 'k')
#    pl.plot(range(1,len(ValuesLLGMMTest)),[-elem for elem in ValuesLLGMMTest[1:len(ValuesLLGMMTest)]],'-rs',markeredgecolor = 'k')
#    pl.yscale('log')
    pl.grid()
    pl.title('log likelihood K-means')
    pl.savefig('Drawings/'+'loglikelihookKmeans.eps')
    pl.show()
    
#    pl.figure()
#    pl.plot(range(len(ValuesLL)),ValuesLL,'-bs',markeredgecolor = 'k')
##    pl.plot(range(len(ValuesLL)),[(-1*elem) for elem in ValuesLL])
##    pl.yscale('log')
#    pl.grid()
#    pl.title('log likelihood K-means')
#    pl.savefig('Drawings/'+'loglikelihookKMEANS.eps')
#    pl.show()
#%%    
    mu1_disp = []
    mu2_disp = []
    mu3_disp = []
    norm_disp = []
    
    for ll in MagnitudesDisplacement_musKmeans:
        mu1_disp.append(ll[0])
        mu2_disp.append(ll[1])
        mu3_disp.append(ll[2])
        norm_disp.append(math.sqrt( pow(ll[0],2) + pow(ll[1],2) + pow(ll[2],2) ))
        
    pl.figure()
    uu1 = pl.plot(range(len(mu1_disp)),mu1_disp,'-r*',markeredgecolor = 'k',label = 'mu_1')
    uu2 = pl.plot(range(len(mu2_disp)),mu2_disp,'-bo',markeredgecolor = 'k',label = 'mu_2')
    uu3 = pl.plot(range(len(mu3_disp)),mu3_disp,'-gs',markeredgecolor = 'k',label = 'mu_3')
    unorm = pl.plot(range(len(norm_disp)),norm_disp,'-m^',markeredgecolor = 'k',label = 'Norm')
    pl.yscale('log')
    pl.grid()
#    pl.legend(handles = [uu1, uu2, uu3])
    pl.legend()
    pl.title('Displacements of means')
    pl.savefig('Drawings/'+'EvolutionMuDisplacement_K_MEANS.eps')
    pl.show()