#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:13:29 2017

@author: Luis Ariel Valbuena
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


#def calculate_alpha(X,P,List_mu,List_CV):
#    alpha = []
#    dimX = P.shape[0]
#    for n in range(len(X)):
#        if(n == 0):
#            temp = []
#            for jj in range(dimX):
#                temp.append(1.0/dimX)
#            alpha.append(temp)
#        else:
#            auxalpha = []
#            for jj in range(dimX):
#                 P_xt_zt_j = hw2.calculate_probability_MultiGaussian([X[n]],List_mu[jj],List_CV[jj])
#                 temp = 0
#                 for ii in range(dimX):
#                     temp = temp + (P[ii,jj]*alpha[n-1][ii])
#                 auxalpha.append(temp*P_xt_zt_j)
#                 ttemp = []
#                 for ee in auxalpha:
#                     ttemp.append(ee/np.sum(auxalpha))
#            alpha.append(ttemp)     
#                 
#    return alpha   
     
def calculate_beta(X,P,List_mu,List_CV):
    beta = []
    for ee in range(len(X)):
        beta.append([])
    dimX = P.shape[0]
    for n in range(len(X)-1,-1,-1):
        if( n == len(X)-1):
            temp = []
            for jj in range(dimX):
                temp.append(1.0/dimX)
            beta[n] = temp
        else:
            auxbeta = []
            for ii in range(dimX):
                temp = 0
                for jj in range(dimX):
                    P_xt_zt_j = hw2.calculate_probability_MultiGaussian([X[n+1]],List_mu[jj],List_CV[jj])
                    temp = temp + (P_xt_zt_j*P[ii,jj]*beta[n+1][jj])#temp = temp + (P_xt_zt_j*P[ii,jj]*beta[n+1][ii])
                auxbeta.append(temp)
                ttemp = []
                for ee in auxbeta:
                    ttemp.append(ee/np.sum(auxbeta))
            beta[n] = ttemp
    return beta

def calculate_alpha_HP(X,P,List_mu,List_CV):
    alpha = []
    dimX = P.shape[0]
    for n in range(len(X)):
        if(n == 0):
            temp = []
            for jj in range(dimX):
                temp.append(1.0/dimX)
            alpha.append(temp)
        else:
            vectorPsi = np.matrix(np.zeros((dimX,1)))
            Alpha_n_minus_1 = np.matrix(np.zeros((dimX,1)))
            for ii in range(dimX):
                vectorPsi[ii,0] = hw2.calculate_probability_MultiGaussian([X[n]],List_mu[ii],List_CV[ii])
                Alpha_n_minus_1[ii,0] = alpha[n-1][ii]
            matrixTemp = (P.T)*Alpha_n_minus_1
            Alpha_n = HadamardProduct(vectorPsi,matrixTemp)
            Alpha_n = Alpha_n/np.sum(Alpha_n)
            temp = Alpha_n.tolist()
            ttemp = []
            for ii in range(dimX):
                ttemp.append(temp[ii][0])
            alpha.append(ttemp)
                 
    return alpha 

def calculate_beta_HP(X,P,List_mu,List_CV):
    beta = []
    for ee in range(len(X)):
        beta.append([])
    dimX = P.shape[0]
    for n in range(len(X)-1,-1,-1):
        if( n == len(X)-1):
            temp = []
            for jj in range(dimX):
                temp.append(1.0/dimX)
            beta[n] = temp
        else:
            vectorPsi = np.matrix(np.zeros((dimX,1)))
            Beta_n_plus_1 = np.matrix(np.zeros((dimX,1)))
            for ii in range(dimX):
                vectorPsi[ii,0] = hw2.calculate_probability_MultiGaussian([X[n+1]],List_mu[ii],List_CV[ii])
                Beta_n_plus_1[ii,0] = beta[n+1][ii]
            matrixTemp = HadamardProduct(vectorPsi,Beta_n_plus_1)
            Beta_n = P*matrixTemp
            Beta_n = Beta_n/np.sum(Beta_n)
            temp = Beta_n.tolist()
            ttemp = []
            for ii in range(dimX):
                ttemp.append(temp[ii][0])
            beta[n] = ttemp
    return beta
            

def HadamardProduct(matrix1,matrix2):
    DimX,DimY = matrix1.shape
    result = np.matrix(np.zeros((DimX,DimY)))
    for ii in range(DimX):
        for jj in range(DimY):
            result[ii,jj] = matrix1[ii,jj]*matrix2[ii,jj]
    return result

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

def calculate_Matrix_xi_t_HP(X,alpha,beta,P,List_mu,List_CV,n):
    dimX = P.shape[0]
    xi_t_tplus1 = np.matrix(np.zeros((dimX,dimX)))
    alpha_n = np.matrix(alpha[n])
    beta_n_plus_1 = np.matrix(beta[n+1])
    vectorPsi_nplus_1 = np.matrix(np.zeros((dimX,1)))
    for ii in range(dimX):
        vectorPsi_nplus_1[ii,0] = hw2.calculate_probability_MultiGaussian([X[n+1]],List_mu[ii],List_CV[ii])
    matrix1 = HadamardProduct(beta_n_plus_1.T,vectorPsi_nplus_1)
    matrix2 = alpha_n.T*matrix1.T
    matrix3 = HadamardProduct(P,matrix2)
#    for ii in range(dimX):
#        xi_t_tplus1[ii,:] = matrix3[ii,:]/np.sum(matrix3[ii,:])
    xi_t_tplus1 = matrix3
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
        
        if(LogLikelihood > LogLikelihood_1)and(iteration <= 20):
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
#            xi_t_tplus1 = calculate_Matrix_xi_t_HP(observed_values,EvolutionAlpha,EvolutionBeta,P,Lista_mu,Lista_sigma,tt)
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

def viterbi(jj,v_pi,List_mu,List_CV,X,P,N):
#    print "jj " + str(kk) + ", n " + str(n) + "len X "+str(len(X))+"\n"
    result = []
    for n in range(N):
        if(n == 0):
            result.append(v_pi[jj]*hw2.calculate_probability_MultiGaussian([X[n]],List_mu[jj],List_CV[jj]))
        else:
            LLista = []
            for ii in range(P.shape[0]):
                temp = hw2.calculate_probability_MultiGaussian([X[n]],List_mu[jj],List_CV[jj])
                LLista.append(P[ii,jj]*temp*result[n-1])
            result.append(np.max(LLista))
    return result

def generateGraphVerification(states_z,EvolutionParameter,title):
    parameter_1 = []
    parameter_2 = []
    parameter_3 = []    
    for parameter_n in EvolutionParameter:
        parameter_1.append(parameter_n[0])
        parameter_2.append(parameter_n[1])
        parameter_3.append(parameter_n[2])
        
    pl.figure()
    N = len(states_z)
#   AspectRatio = 2.5
    pl.subplot(411)
    min_x = 0
    max_x = N +1
    min_y = np.min(real_states) - 0.5
    max_y = np.max(real_states) + 1.5
    axesList = [min_x, max_x, min_y, max_y]
    markerline, stemlines, baseline = pl.stem(range(1,len(states_z)+1), [zz+1 for zz in states_z] , '-')
    pl.setp(baseline, 'color', 'r', 'linewidth', 2)
    pl.grid()
#   ax = pl.gca()
    pl.axis(axesList)
#   ax.set_aspect(AspectRatio)
    pl.yticks(np.arange(0, 4, 1.0))
    pl.xticks(np.arange(0, N + 10, 10))
    pl.title("Observed variable Z")    
    
#    AspectRatio = 100
    pl.subplot(412)    
    min_x = 0
    max_x = N + 1
    min_y = -0.05
    max_y =  1.2
    axesList = [min_x, max_x, min_y, max_y]
    
    markerline, stemlines, baseline = pl.stem(range(1,len(parameter_1)+1), parameter_1, '-')
    pl.setp(baseline, 'color', 'r', 'linewidth', 2)
    pl.grid()
#    ax = pl.gca()
    pl.axis(axesList)
#   ax.set_aspect(AspectRatio)
    pl.xticks(np.arange(0, N + 10, 10))
    pl.title(title + "_{1}")

    pl.subplot(413)  
    markerline, stemlines, baseline = pl.stem(range(1,len(parameter_2)+1), parameter_2, '-')
    pl.setp(baseline, 'color', 'r', 'linewidth', 2)
    pl.grid()
#   ax = pl.gca()
    pl.axis(axesList)
#   ax.set_aspect(AspectRatio)
    pl.xticks(np.arange(0, N + 10, 10))
    pl.title(title + "_{2}")
    
    pl.subplot(414)
    markerline, stemlines, baseline = pl.stem(range(1,len(parameter_3)+1), parameter_3, '-')
    pl.setp(baseline, 'color', 'r', 'linewidth', 2)
    pl.grid()
#   ax = pl.gca()
    pl.axis(axesList)
#   ax.set_aspect(AspectRatio)
    pl.xticks(np.arange(0, N + 10, 10))
    pl.title(title + "_{3}")
    
    pl.subplots_adjust(top=0.94, bottom=0.08, left=0.05, right=0.95, hspace=1.15, wspace=0.35)
    
    pl.savefig('Drawings/'+'real States vs '+title+'.eps')
    pl.show()
                
                

if __name__ == "__main__":
    
    #Parameters for the artificial generated data
    P_generation = np.matrix([[0.8, 0.1, 0.1],[0.2, 0.5, 0.3],[0.3, 0.1, 0.6]])
    list_mu = [1, 2, 3]
    list_sigma = [1/3.0, 1/3.0, 1/3.0]
    N = 100
    NRealizations = 80
    AspectRatio = 12

    X = []
    Z = []
    for ii in range(NRealizations):
        observed_values , real_states = markovProcess(P_generation,list_sigma,list_mu,N)
        X.append(observed_values)
        Z.append(real_states)
    
    targetIndex = 54
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
            
    Lista_mu = [np.matrix([[1.5]]), np.matrix([[2.2]]), np.matrix([[3.8]])]
    Lista_sigma = [np.matrix([[1.0]]), np.matrix([[1.0]]), np.matrix([[1.0]])]
    v_pi = [1.0/len(Lista_mu), 1.0/len(Lista_mu), 1.0/len(Lista_mu)]       
    
    concatenatedData = []
    for arrayy in X:
        concatenatedData = np.concatenate((concatenatedData, arrayy))
    
    XTrain, XTest = get_Train_Test_Data(concatenatedData,0.85)             
    Lista_mu,Lista_sigma,v_pi = EM_GMM(XTrain,XTest,Lista_mu,Lista_sigma,v_pi,N_States)

    arrayLista_mu.append(Lista_mu)
    arrayLista_sigma.append(Lista_sigma)
    arrayv_pi.append(v_pi)

    estimation_P = np.matrix(P)
    
    Traj_Lista_mu = []
    Traj_Lista_sigma = []
    Traj_Lista_v_pi = []
    Traj_estimationP = []
    
    Traj_Lista_mu.append(Lista_mu)
    Traj_Lista_sigma.append(Lista_sigma)
    Traj_Lista_v_pi.append(v_pi)
#%%    
    for iterator in range(15):
        print "\nMain iterator " + str(iterator)
        arrayEvolAlpha = []
        arrayEvolBeta = []
        arrayEvolForwBack = []        
        for ii in range(len(X)):
            observed_values = X[ii]
            
            #E Step
            #Testing alpha, beta, forwards-backwards
#            EvolutionAlpha = calculate_alpha(observed_values,estimation_P,Lista_mu,Lista_sigma)
            EvolutionAlpha = calculate_alpha_HP(observed_values,estimation_P,Lista_mu,Lista_sigma)
            EvolutionBeta = calculate_beta(observed_values,estimation_P,Lista_mu,Lista_sigma)
#            EvolutionBeta = calculate_beta_HP(observed_values,estimation_P,Lista_mu,Lista_sigma)
            EvolutionForwBack = calculate_Forwards_Backwards(EvolutionAlpha,EvolutionBeta)
            
            arrayEvolAlpha.append(EvolutionAlpha)
            arrayEvolBeta.append(EvolutionBeta)
            arrayEvolForwBack.append(EvolutionForwBack)
        
            if(ii == targetIndex):
                generateGraphVerification(real_states,EvolutionAlpha,'Alpha_iteration '+str(iterator))
                generateGraphVerification(real_states,EvolutionBeta,'Beta_iteration '+str(iterator))
                generateGraphVerification(real_states,EvolutionForwBack,'Gamma_iteration '+str(iterator))

        
        #M Step
        #Estimating matrix P

        estimation_P = estimate_P(X,arrayEvolAlpha,arrayEvolBeta,N_States)
        E_N_1k = np.matrix(np.zeros((N_States,1)))
        for array_gamma1 in arrayEvolForwBack[0]:
            for kk in range(N_States):
                E_N_1k[kk,0] = E_N_1k[kk,0] + array_gamma1[kk]
#        E_N_1k =  E_N_1k/NRealizations 
        E_N_1k =  E_N_1k/N
        v_pi = []
        for nn in range(N_States):
            v_pi.append(E_N_1k[nn,0])
        
        E_N_j = np.matrix(np.zeros((N_States,1)))
        for ii in range(NRealizations):
            for t in range(N):
                for kk in range(N_States):
                    E_N_j[kk,0] = E_N_j[kk,0] + arrayEvolForwBack[ii][t][kk]
        
        E_bar_x_k = np.matrix(np.zeros((N_States,1)))
        E_bar_xx_k_T = np.matrix(np.zeros((N_States,1)))
        for ii in range(NRealizations):
            for t in range(N):
                for kk in range(N_States):
                    temp = arrayEvolForwBack[ii][t][kk]*X[ii][t]
                    E_bar_x_k[kk,0] = E_bar_x_k[kk,0] + temp
                    E_bar_xx_k_T[kk,0] = E_bar_xx_k_T[kk,0] + X[ii][t]*temp
                                
        Lista_mu = []
        Lista_sigma = []
        
        for kk in range(N_States):
            Lista_mu.append( np.matrix([[ E_bar_x_k[kk,0]/E_N_j[kk,0] ]]) )
            Lista_sigma.append( np.matrix( (1/E_N_j[kk,0])*(E_bar_xx_k_T[kk,0] - (E_N_j[kk,0]*Lista_mu[-1]*Lista_mu[-1]) ) ) )
        
        print "\nestimation_P"
        print estimation_P
        
        print "\nLista mu"
        print Lista_mu
        
        print "\nLista sigma"
        print Lista_sigma
        
        print "\nv_pi"
        print v_pi
            
        print "\nP_generation"
        print P_generation
        
        Traj_Lista_mu.append(Lista_mu)
        Traj_Lista_sigma.append(Lista_sigma)
        Traj_Lista_v_pi.append(v_pi)
        Traj_estimationP.append(estimation_P)
    
    print "\nFinal\n"
    print "Traj Lista_mu"
    for arrray in Traj_Lista_mu:
        print arrray
    
    print "\nTraj Lista_sigma"
    for arrray in Traj_Lista_sigma:
        print arrray
    
    print "\nTraj Lista_v_pi"
    print Traj_Lista_v_pi
    
    print "\nTraj_estimationP"
    for matt in Traj_estimationP:
        print "\n"
        print matt
#%%        (jj,v_pi,List_mu,List_CV,X,P,n)
    Linea1 = []
    Linea2 = []
    Linea3 = []
    realizationSeq = X[targetIndex]
        
    Linea1.append(viterbi(0,v_pi,Lista_mu,Lista_sigma,realizationSeq,P_generation,N))
    Linea2.append(viterbi(1,v_pi,Lista_mu,Lista_sigma,realizationSeq,P_generation,N))
    Linea3.append(viterbi(2,v_pi,Lista_mu,Lista_sigma,realizationSeq,P_generation,N))
