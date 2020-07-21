# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:10:12 2019

@author: sp3215
"""
import numpy as np


#################################################
# DEFINE FUNCTION: H CONTRACTIVE DERIVATIVE
##################################################
def Hc1_con(rho,pot,theta): 
    if pot==1:# Double well
        Hc1=rho**3
    elif pot==2 and theta!=0:# Logarithmic
        Hc1=theta/2.*np.log(np.divide(1+rho,1-rho))
        greater1_index = np.append(rho,0) > 1
        lowerminus1_index = np.append(rho,0) < -1
        Hc1[greater1_index[:-1]]=9999999999999999999999999
        Hc1[lowerminus1_index[:-1]]=-9999999999999999999999     
    else:
        Hc1=np.zeros(np.shape(rho))    
    return Hc1

####################################################
# DEFINE FUNCTION: H CONTRACTIVE 2 DERIVATIVE 
#####################################################3
def Hc2_con(rho,pot,theta):
    if pot==1:# Double well
        Hc2=3*rho**2
    elif pot==2 and theta!=0:# Logarithmic
        if np.abs(rho)<=1:        
            Hc2=theta/(1-rho**2)
        else:
            Hc2=np.inf            
    else:
        Hc2=np.zeros(np.shape(rho))
    return Hc2

###############################################
# DEFINE FUNCTION: H EXPANSIVE 2 DERIVATIVE
###############################################    
def He1_exp(rho,pot,thetac):
    if pot==1:# Double well
        He1=rho
    elif pot==2 and thetac!=0:# Logarithmic
        He1=thetac*rho
    else:
        He1=np.zeros(np.shape(rho))
    return He1

##############################
# DEFINE FUNCTION: MOBILITY
##################################
def mobility(rho,choicemob): 
    if choicemob==1:
        m=np.zeros(np.shape(rho))
        m[:]=1
    elif choicemob==2:
        m=1-rho**2
    return m

 

