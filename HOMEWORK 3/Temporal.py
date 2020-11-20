#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 23:45:36 2017

@author: ariel
"""

#%%

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
import matplotlib.colors as mplc
import numpy as np
#from svmutil import *
import math
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
    AspectRatio = 2.5
#    pl.subplot(411)
    min_x = 0
    max_x = N +1
    min_y = np.min(real_states) - 0.5
    max_y = np.max(real_states) + 1.5
    axesList = [min_x, max_x, min_y, max_y]
    markerline, stemlines, baseline = pl.stem(range(1,len(states_z)+1), [zz+1 for zz in states_z] , '-')
    pl.setp(baseline, 'color', 'r', 'linewidth', 2)
    pl.grid()
    ax = pl.gca()
    pl.axis(axesList)
    ax.set_aspect(AspectRatio)
    pl.yticks(np.arange(0, 4, 1.0))
    pl.xticks(np.arange(0, N + 10, 10))
    pl.title("Observed variable Z")    
    
#    AspectRatio = 100
#    pl.subplot(412) 
    AspectRatio = 30
    pl.figure() 
    min_x = 0
    max_x = N + 1
    min_y = -0.05
    max_y =  1.0
    axesList = [min_x, max_x, min_y, max_y]
    
    markerline, stemlines, baseline = pl.stem(range(1,len(parameter_1)+1), parameter_1, '-')
    pl.setp(baseline, 'color', 'r', 'linewidth', 2)
    pl.grid()
    ax = pl.gca()
    pl.axis(axesList)
    ax.set_aspect(AspectRatio)
    pl.xticks(np.arange(0, N + 10, 10))
    pl.title(title + "(1)")
    pl.savefig('Drawings2/'+title+'(1).eps', bbox_inches='tight')

#    pl.subplot(413) 
    pl.figure()  
    markerline, stemlines, baseline = pl.stem(range(1,len(parameter_2)+1), parameter_2, '-')
    pl.setp(baseline, 'color', 'r', 'linewidth', 2)
    pl.grid()
    ax = pl.gca()
    pl.axis(axesList)
    ax.set_aspect(AspectRatio)
    pl.xticks(np.arange(0, N + 10, 10))
    pl.title(title + "(2)")
    pl.savefig('Drawings2/'+title+'(2).eps', bbox_inches='tight')
    
#    pl.subplot(414)
    pl.figure() 
    markerline, stemlines, baseline = pl.stem(range(1,len(parameter_3)+1), parameter_3, '-')
    pl.setp(baseline, 'color', 'r', 'linewidth', 2)
    pl.grid()
    ax = pl.gca()
    pl.axis(axesList)
    ax.set_aspect(AspectRatio)
    pl.xticks(np.arange(0, N + 10, 10))
    pl.title(title + "(3)")
    pl.savefig('Drawings2/'+title+'(3).eps', bbox_inches='tight')
    
#    pl.subplots_adjust(top=0.94, bottom=0.08, left=0.05, right=0.95, hspace=1.15, wspace=0.35)
    
#    pl.savefig('Drawings2/'+'real States vs '+title+'.eps')
    pl.show()
    
if __name__ == "__main__":
    SuperIndex = 54
    pl.figure() 
    min_x = -1
    max_x = N + 1
    min_y = np.min(obs_vals) - 0.5
    max_y = np.max(obs_vals) + 0.5
    axesList = [min_x, max_x, min_y, max_y]    
    markerline, stemlines, baseline = pl.stem(range(1,len(X[SuperIndex])+1), X[SuperIndex] , '-')
    pl.setp(baseline, 'color', 'r', 'linewidth', 2)
    pl.grid()
    ax = pl.gca()
    pl.axis(axesList)
    ax.set_aspect(AspectRatio)
    pl.xticks(np.arange(0, N + 10, 10))
    pl.title("Observed variable X")
    pl.savefig('Drawings2/'+'ObservedvariableX.eps', bbox_inches='tight')
    pl.show()
    
    
    #Drawing the originial states
    pl.figure()
    min_x = -1
    max_x = N + 1
    min_y = np.min(real_states) - 0.5
    max_y = np.max(real_states) + 1.5
    axesList = [min_x, max_x, min_y, max_y]
    markerline, stemlines, baseline = pl.stem(range(1,len(Z[SuperIndex])+1), [zz+1 for zz in Z[SuperIndex]] , '-')
    pl.setp(baseline, 'color', 'r', 'linewidth', 2)
    pl.grid()
    ax = pl.gca()
    pl.axis(axesList)
    ax.set_aspect(AspectRatio)
    pl.title("Observed variable Z")
    pl.xticks(np.arange(0, N + 10, 10))
    pl.savefig('Drawings2/'+'ObservedvariableZ.eps', bbox_inches='tight')
    pl.show()
    
    EvolutionAlphaF = arrayEvolAlpha[SuperIndex]
    EvolutionBetaF = arrayEvolBeta[SuperIndex]
    EvolutionForwBackF = arrayEvolForwBack[SuperIndex]
    iterator = 3
    generateGraphVerification(Z[SuperIndex],EvolutionAlphaF,'Alpha')
    generateGraphVerification(Z[SuperIndex],EvolutionBetaF,'Beta')
    generateGraphVerification(Z[SuperIndex],EvolutionForwBackF,'Gamma')
#%%
    traj_mu_1 = []
    traj_mu_2 = []
    traj_mu_3 = []
    traj_sigma_1 = []
    traj_sigma_2 = []
    traj_sigma_3 = []
    
    for elem_mu, elem_sigma in zip(Traj_Lista_mu,Traj_Lista_sigma):
        traj_mu_1.append(elem_mu[0].tolist()[0][0])
        traj_mu_2.append(elem_mu[1].tolist()[0][0])
        traj_mu_3.append(elem_mu[2].tolist()[0][0])
        traj_sigma_1.append(elem_sigma[0].tolist()[0][0])
        traj_sigma_2.append(elem_sigma[1].tolist()[0][0])
        traj_sigma_3.append(elem_sigma[2].tolist()[0][0])
        
    pl.figure()
    pl.plot(range(1,len(traj_mu_1)+1),traj_mu_1 ,'-rs',markeredgecolor = 'k') 
    pl.plot(range(1,len(traj_mu_2)+1),traj_mu_2 ,'-bo',markeredgecolor = 'k') 
    pl.plot(range(1,len(traj_mu_3)+1),traj_mu_3 ,'-g*',markeredgecolor = 'k')
    pl.xlabel("iteration")
    pl.xticks(np.arange(0,len(traj_mu_3)+1,1))
    pl.title("Evolution means")
#    pl.yscale('log')
    pl.grid()
    pl.savefig('Drawings2/'+'EvolutionMeans.eps', bbox_inches='tight')
    
    pl.figure()
    pl.plot(range(1,len(traj_sigma_1)+1),traj_sigma_1 ,'-rs',markeredgecolor = 'k') 
    pl.plot(range(1,len(traj_sigma_2)+1),traj_sigma_2 ,'-bo',markeredgecolor = 'k') 
    pl.plot(range(1,len(traj_sigma_3)+1),traj_sigma_3 ,'-g*',markeredgecolor = 'k')
    pl.xlabel("iteration")
    pl.xticks(np.arange(0,len(traj_mu_3)+1,1))
    pl.title("Evolution sigma")
    pl.grid() 
    pl.savefig('Drawings2/'+'EvolutionSigmas.eps', bbox_inches='tight')   