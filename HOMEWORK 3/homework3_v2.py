#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 22:57:02 2017

@author: Luis Ariel Valbuena Reyes
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
import matplotlib.colors as mplc
import numpy as np
#from svmutil import *
import math

import HW2 as hw2

def markovProcess(P,list_sigma,list_mu,N):
    p = np.cumsum(P,axis=1)
    zact = np.ceil( np.random.uniform()*len(list_mu) ) - 1 #Maybe we need to subtract -1

    z = []
    for ii in range(N):
        a = np.random.uniform()
        auxMatrix = p[zact,:] > a
        temp = p[zact,:].tolist()
        b = 0
        temp = auxMatrix.tolist()
        listaBooleans = []
        for element in temp[0]:
            if (element == True):
                listaBooleans.append(b)
            b = b + 1
        if listaBooleans:
            zact = np.min(listaBooleans)
            z.append(zact)
    x = np.zeros(len(z))
    distribution = np.random.normal(size = len(z))
    c = 0
    ret_z = []
    for zz in z:
        x[c] = distribution[c]*list_sigma[zz] + list_mu[zz]
        c = c + 1
        ret_z.append(zz+1)
    return (x,z)
        

def estimateTransitionProbabilities(z):
    #with indexing starting from 0
    N_states = np.max(z) + 1
    P_est = np.matrix(np.zeros((N_states,N_states)))
    
    N = len(z)
    for ii in range(N-1):
        P_est[z[ii],z[ii+1]] =  P_est[z[ii],z[ii+1]] + 1
    
    DimX, DimY = P_est.shape
    
    for ii in range(DimX):
        summation = np.sum(P_est[ii,:])
        P_est[ii,:] = (1.0/summation)*P_est[ii,:]
    
    return P_est


def calculate_alpha(X,P,List_mu,List_CV):
    alpha = []
    dimX = P.shape[0]
    for n in range(len(X)):
        if(n == 0):
            temp = []
            for jj in range(dimX):
                temp.append(1.0/dimX)
            alpha.append(temp)
        else:
            auxalpha = []
            for jj in range(dimX):
                 P_xt_zt_j = hw2.calculate_probability_MultiGaussian([X[n]],List_mu[jj],List_CV[jj])
                 temp = 0
                 for ii in range(dimX):
                     temp = temp + (P[ii,jj]*alpha[n-1][ii])
                 auxalpha.append(temp*P_xt_zt_j)
                 ttemp = []
                 for ee in auxalpha:
                     ttemp.append(ee/np.sum(auxalpha))
            alpha.append(ttemp)     
                 
    return alpha        


def calculate_beta(X,P,List_mu,List_CV):
    beta = []
    for ee in range(len(X)):
        beta.append([])
    dimX = P.shape[0]
    for n in range(len(X)-1,-1,-1):
        if( n == len(X)-1):
            temp = []
            for jj in range(dimX):
                temp.append(1.0)
            beta[n] = temp
        else:
            auxbeta = []
            for ii in range(dimX):
                temp = 0
                for jj in range(dimX):
                    P_xt_zt_j = hw2.calculate_probability_MultiGaussian([X[n]],List_mu[jj],List_CV[jj])
                    temp = temp + (P_xt_zt_j*P[ii,jj]*beta[n+1][jj])#temp = temp + (P_xt_zt_j*P[ii,jj]*beta[n+1][ii])
                auxbeta.append(temp)
            beta[n] = auxbeta
            
    return beta
            

def calculate_Forwards_Backwards(alpha,beta):
    gamma = []
    for vect_alpha, vect_beta in zip(alpha,beta):
        temp = []
        for alpha_i,beta_i in zip(vect_alpha,vect_beta):
            temp.append(alpha_i*beta_i)
        ttemp = []
        for ee in temp:
            ttemp.append(ee/np.sum(temp))
        gamma.append(ttemp)
        
    return gamma


def calculate_Matrix_xi_t(X,alpha,beta,P,List_mu,List_CV,n):
    dimX = P.shape[0]
    xi_t_tplus1 = np.matrix(np.zeros((dimX,dimX)))
    for i in range(dimX):
        for j in range(dimX):
            alpha_t_i = alpha[n][i]
            beta_tplus1_j = beta[n+1][j]
            P_ij = P[i,j]
            P_xt_zt_j = hw2.calculate_probability_MultiGaussian([X[n+1]],List_mu[j],List_CV[j])
            xi_t_tplus1[i,j] = alpha_t_i*beta_tplus1_j*P_ij*P_xt_zt_j
        xi_t_tplus1[i,:] = xi_t_tplus1[i,:]/np.sum(xi_t_tplus1[i,:])
    return xi_t_tplus1
    

def get_Train_Test_Data(X,threshold):
    Temp1 = []
    Temp2 = []
    for elem in X.tolist():
        if (np.random.uniform()<= threshold):
            Temp1.append(np.matrix([[elem]]))
        else:
            Temp2.append(np.matrix([[elem]]))
    if(len(Temp1)>=len(Temp2)):
        XTrain = list(Temp1)
        XTest = list(Temp2)
    else:
        XTrain = list(Temp2)
        XTest = list(Temp1)
    
    return (XTrain,XTest)


def EM_GMM(XTrain,XTest,Lista_mu0,Lista_sigma0,v_pi0,N_States):
    LogLikelihood_1 = -9999999999
    iteration = 1
    Lista_mu = list(Lista_mu0) 
    Lista_sigma = list(Lista_sigma0)
    v_pi = list(v_pi0)
    while(True):
        print "\n\tIteration GMM " + str(iteration)
        resultUpdate = hw2.updateParameters(v_pi, XTrain,Lista_mu,Lista_sigma,N_States)
        v_pi = resultUpdate[0]
        Lista_mu = resultUpdate[1]

        Lista_sigma = []
        for CVm in resultUpdate[2]:
            Lista_sigma.append(np.matrix(CVm))

        LogLikelihood = hw2.calculate_LogLikelihood(Lista_mu,Lista_sigma,v_pi,XTest)
        
        iteration = iteration + 1
        
        if(LogLikelihood > LogLikelihood_1)and(iteration <= 30):
            LogLikelihood_1 = LogLikelihood 
        else:
            break
        
    print "\nmus"
    print Lista_mu
        
    return (Lista_mu,Lista_sigma,v_pi)


def estimate_P(X,arrayEvolAlpha,arrayEvolBeta,N_States):
    Evolution_Ximatrices = []
    estimation_P =np.matrix(np.zeros((N_States,N_States)))
    denominator = np.matrix(np.zeros((N_States,1)))
    
    summation_on_Time = []
    for ii in range(len(X)):
        observed_values = X[ii]
        EvolutionAlpha = arrayEvolAlpha[ii]
        EvolutionBeta = arrayEvolBeta[ii]
        Xi_matrices = []
        temp = np.matrix(np.zeros((N_States,N_States)))
        for tt in range(N-1):
            xi_t_tplus1 = calculate_Matrix_xi_t(observed_values,EvolutionAlpha,EvolutionBeta,P,Lista_mu,Lista_sigma,tt)
            Xi_matrices.append(xi_t_tplus1)
            temp = temp + xi_t_tplus1
#            estimation_P = estimation_P + Xi_matrices[tt]
#            denominator = denominator + np.matrix(EvolutionForwBack[tt]).T
        summation_on_Time.append(temp)
        Evolution_Ximatrices.append(Xi_matrices)

    summationTotal = np.matrix(np.zeros((N_States,N_States)))
     
    for Xi_i in summation_on_Time:
        summationTotal = summationTotal + Xi_i

    denominator = summationTotal[:,0]
    for kk in range(1,N_States):
        denominator = denominator + summationTotal[:,kk]
        
    for kk in range(N_States):
        estimation_P[kk,:] = summationTotal[kk,:]/denominator[kk,0]
        
    return estimation_P


if __name__ == "__main__":
    
    #Parameters for the artificial generated data
    P_generation = np.matrix([[0.8, 0.1, 0.1],[0.2, 0.5, 0.3],[0.3, 0.1, 0.6]])
    list_mu = [1, 2, 3]
    list_sigma = [1/3.0, 1/3.0, 1/3.0]
    N = 100
    NRealizations = 2
    AspectRatio = 12

    X = []
    Z = []
    for ii in range(NRealizations):
        observed_values , real_states = markovProcess(P_generation,list_sigma,list_mu,N)
        X.append(observed_values)
        Z.append(real_states)
    
    targetIndex = 0
    obs_vals = X[targetIndex]
    real_states = Z[targetIndex]
    
    #Drawing generated artificial data
    pl.figure() 
    min_x = -1
    max_x = N + 1
    min_y = np.min(obs_vals) - 0.5
    max_y = np.max(obs_vals) + 0.5
    axesList = [min_x, max_x, min_y, max_y]    
    markerline, stemlines, baseline = pl.stem(range(1,len(obs_vals)+1), obs_vals , '-')
    pl.setp(baseline, 'color', 'r', 'linewidth', 2)
    pl.grid()
    ax = pl.gca()
    pl.axis(axesList)
    ax.set_aspect(AspectRatio)
    pl.title("Observed variable X")
    pl.show()
    
    
    #Drawing the originial states
    pl.figure()
    min_x = -1
    max_x = N + 1
    min_y = np.min(real_states) - 0.5
    max_y = np.max(real_states) + 1.5
    axesList = [min_x, max_x, min_y, max_y]
    markerline, stemlines, baseline = pl.stem(range(1,len(real_states)+1), [zz+1 for zz in real_states] , '-')
    pl.setp(baseline, 'color', 'r', 'linewidth', 2)
    pl.grid()
    ax = pl.gca()
    pl.axis(axesList)
    ax.set_aspect(AspectRatio)
    pl.title("Observed variable Z")
    pl.show()
      
#    print "P_generation"
#    print P_generation
#    print "P__est"
#    print P__est
#    
#    print "Different"
#    print P_generation - P__est

#    pl.figure()
#    pl.hist(observed_values,bins='auto')
#    pl.grid()    
    

    #Initialization
    
    N_States = 3
    
    #Required estimation requested on item 2.
    P__est = np.matrix(np.zeros((N_States,N_States)))
    for chain_z in Z:
        P__est = P__est + estimateTransitionProbabilities(chain_z)  
    P__est = P__est/len(Z)
    
#    alpha = []
#    beta = []
#    for ii in range(N_States):
#        alpha.append(1.0/N_States)
#        beta.append(1.0)
          
    P = np.matrix(np.random.uniform(size=(N_States,N_States)))  
    for ii in range(N_States):
        P[ii,:] = P[ii,:]/np.sum(P[ii,:])
    
    #generate trial and test 
#    cc = 1
#    XTrain = []
#    XTest = []
#    for elem in observed_values_1.tolist():
#        if (cc % 2 == 1):
#            XTrain.append(np.matrix([[elem]]))
#        else:
#            XTest.append(np.matrix([[elem]]))
    arrayLista_mu = []
    arrayLista_sigma = []
    arrayv_pi = []
    arrayEvolAlpha = []
    arrayEvolBeta = []
    arrayEvolForwBack = []
    
    for ii in range(len(X)):
        print "\nmain iteration ii " + str(ii)
        observed_values = X[ii]
        if (ii<6):        
            Lista_mu = [np.matrix([[1.5]]), np.matrix([[2.2]]), np.matrix([[3.8]])]
            Lista_sigma = [np.matrix([[1.0]]), np.matrix([[1.0]]), np.matrix([[1.0]])]
            v_pi = [1.0/len(Lista_mu), 1.0/len(Lista_mu), 1.0/len(Lista_mu)]       
                
            XTrain, XTest = get_Train_Test_Data(observed_values,0.51)             
            Lista_mu,Lista_sigma,v_pi = EM_GMM(XTrain,XTest,Lista_mu,Lista_sigma,v_pi,N_States)
        
            arrayLista_mu.append(Lista_mu)
            arrayLista_sigma.append(Lista_sigma)
            arrayv_pi.append(v_pi)
        
        #Testing alpha, beta, forwards-backwards
        EvolutionAlpha = calculate_alpha(observed_values,P,Lista_mu,Lista_sigma)
        EvolutionBeta = calculate_beta(observed_values,P,Lista_mu,Lista_sigma)
        EvolutionForwBack = calculate_Forwards_Backwards(EvolutionAlpha,EvolutionBeta)
        
        arrayEvolAlpha.append(EvolutionAlpha)
        arrayEvolBeta.append(EvolutionBeta)
        arrayEvolForwBack.append(EvolutionForwBack)
    

        
        if(ii == targetIndex):
            gamma_1 = []
            gamma_2 = []
            gamma_3 = []    
            for gamma_n in EvolutionForwBack:
                gamma_1.append(gamma_n[0])
                gamma_2.append(gamma_n[1])
                gamma_3.append(gamma_n[2])
                
            pl.figure()
            AspectRatio = 2.5
            pl.subplot(411)
            min_x = 0
            max_x = N +1
            min_y = np.min(real_states) - 0.5
            max_y = np.max(real_states) + 1.5
            axesList = [min_x, max_x, min_y, max_y]
            markerline, stemlines, baseline = pl.stem(range(1,len(real_states)+1), [zz+1 for zz in real_states] , '-')
            pl.setp(baseline, 'color', 'r', 'linewidth', 2)
            pl.grid()
            ax = pl.gca()
            pl.axis(axesList)
#            ax.set_aspect(AspectRatio)
            pl.yticks(np.arange(0, 4, 1.0))
            pl.xticks(np.arange(0, N + 10, 10))
            pl.title("Observed variable Z")    
            
            AspectRatio = 100
            pl.subplot(412)    
            min_x = 0
            max_x = N + 1
            min_y = -0.05
            max_y =  1.2
            axesList = [min_x, max_x, min_y, max_y]
            
            markerline, stemlines, baseline = pl.stem(range(1,len(gamma_1)+1), gamma_1, '-')
            pl.setp(baseline, 'color', 'r', 'linewidth', 2)
            pl.grid()
            ax = pl.gca()
            pl.axis(axesList)
#            ax.set_aspect(AspectRatio)
            pl.xticks(np.arange(0, N + 10, 10))
            pl.title("\gamma_{1}")

            pl.subplot(413)  
            markerline, stemlines, baseline = pl.stem(range(1,len(gamma_2)+1), gamma_2, '-')
            pl.setp(baseline, 'color', 'r', 'linewidth', 2)
            pl.grid()
            ax = pl.gca()
            pl.axis(axesList)
#            ax.set_aspect(AspectRatio)
            pl.xticks(np.arange(0, N + 10, 10))
            pl.title("\gamma_{2}")
            
            pl.subplot(414)
            markerline, stemlines, baseline = pl.stem(range(1,len(gamma_3)+1), gamma_3, '-')
            pl.setp(baseline, 'color', 'r', 'linewidth', 2)
            pl.grid()
            ax = pl.gca()
            pl.axis(axesList)
#            ax.set_aspect(AspectRatio)
            pl.xticks(np.arange(0, N + 10, 10))
            pl.title("\gamma_{3}")
            
            pl.subplots_adjust(top=0.94, bottom=0.08, left=0.05, right=0.95, hspace=1.15, wspace=0.35)
            
            pl.savefig('Drawings/'+'real States vs Gamma.eps')
            pl.show()


            print "Lista mu"
            print Lista_mu
            print "\n"    
            
            print "Lista sigma"
            print Lista_sigma
            print "\n"   
    
#%%    
    #Estimating matrix P
#    Evolution_Ximatrices = []
#    estimation_P =np.matrix(np.zeros((N_States,N_States)))
#    denominator = np.matrix(np.zeros((N_States,1)))
#    
#    summation_on_Time = []
#    for ii in range(len(X)):
#        observed_values = X[ii]
#        EvolutionAlpha = arrayEvolAlpha[ii]
#        EvolutionBeta = arrayEvolBeta[ii]
#        Xi_matrices = []
#        temp = np.matrix(np.zeros((N_States,N_States)))
#        for tt in range(N-1):
#            xi_t_tplus1 = calculate_Matrix_xi_t(observed_values,EvolutionAlpha,EvolutionBeta,P,Lista_mu,Lista_sigma,tt)
#            Xi_matrices.append(xi_t_tplus1)
#            temp = temp + xi_t_tplus1
##            estimation_P = estimation_P + Xi_matrices[tt]
##            denominator = denominator + np.matrix(EvolutionForwBack[tt]).T
#        summation_on_Time.append(temp)
#        Evolution_Ximatrices.append(Xi_matrices)
#
#    summationTotal = np.matrix(np.zeros((N_States,N_States)))
#     
#    for Xi_i in summation_on_Time:
#        summationTotal = summationTotal + Xi_i
#
#    denominator = summationTotal[:,0]
#    for kk in range(1,N_States):
#        denominator = denominator + summationTotal[:,kk]
#        
#    for kk in range(N_States):
#        estimation_P[kk,:] = summationTotal[kk,:]/denominator[kk,0]
    estimation_P = estimate_P(X,arrayEvolAlpha,arrayEvolBeta,N_States)
    print "\nestimation_P"
    print estimation_P
        
    print "\nP_generation"
    print P_generation
    

        
        
    
