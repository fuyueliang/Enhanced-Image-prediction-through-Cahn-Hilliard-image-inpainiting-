# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:10:12 2019

@author: sp3215
"""
import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.image as mpimg
# from skimage import io
# import os
# dirpath = os.getcwd()
# import sys
# sys.path.append(dirpath+'\\functions')
# from test import CH_image_final
# from test import i
# from test import ntimes
# from test import rho




##################
# DEFINE FUNCTION: H CONTRACTIVE DERIVATIVE
##################
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

##################
# DEFINE FUNCTION: H CONTRACTIVE 2 DERIVATIVE 
##################
def Hc2_con(rho,pot,theta):
    if pot==1:# Double well
        Hc2=3*rho**2
    elif pot==2 and theta!=0:# Logarithmic
        if np.abs(rho)<=1:        
            Hc2=theta/(1-rho**2)
        else:
            Hc2=np.inf            
#        greaterabs1_index = np.append(np.abs(rho),0) > 1
#        Hc2[greaterabs1_index[:-1]]=99999999999
    else:
        Hc2=np.zeros(np.shape(rho))
    return Hc2

##################
# DEFINE FUNCTION: H EXPANSIVE 2 DERIVATIVE
##################    
def He1_exp(rho,pot,thetac):
    if pot==1:# Double well
        He1=rho
    elif pot==2 and thetac!=0:# Logarithmic
        He1=thetac*rho
#        greater1_index = np.append(rho,0) > 1
#        lowerminus1_index = np.append(rho,0) < -1
#        He1[greater1_index[:-1]]=-9999
#        He1[lowerminus1_index[:-1]]=+9999
    else:
        He1=np.zeros(np.shape(rho))
    return He1

##################
# DEFINE FUNCTION: MOBILITY
##################
def mobility(rho,choicemob): 
    if choicemob==1:
        m=np.zeros(np.shape(rho))
        m[:]=1
    elif choicemob==2:
        m=1-rho**2
    return m

##################
# DEFINE FUNCTION: Transpose reshape
##################
# def tr(u,n,choice): 
#     output=np.reshape(np.transpose(np.reshape(Hc1, (n,n))),n*n)  
#     if choice==1:
#         return output[1:]
#     elif choice==-1:
#         return output[0:-1]
  



##################
# DEFINE FUNCTION: FLUX 
##################


# def Euler_implicit_1(rho,rhon,n,dx,dt,epsilon,ntimes,pot,theta,thetac,choicemob):
    
#     # Create matrix rho
    
#     rho=np.reshape(rho,(n,n))
#     rhon=np.reshape(rhon,(n,n)) # rho in the future
    
#     # Define variation of free energy
    
#     # a) Hc: contractive (convex) part of the free energy (treated implicitly))
#     # Hc1 is the first derivative of Hc
    
#     Hc1= Hc1_con(rhon,pot,theta)
    
#     # b) He: expansive (concave) part of the free energy (treated explicitly)
#     # He1 is the first derivative of He    
    
#     He1= He1_exp(rho,pot,thetac)
    
#     # c) Laplacian (treated semi-implicitly)
#     #see the difference for cells in the middle, at the corner, lying on the line

#     Lap=np.zeros((n,n))
    
#     Lap[1:-1,1:-1]=epsilon**2*(-4*rho[1:-1,1:-1]+rho[0:-2,1:-1]+rho[2:,1:-1]+rho[1:-1,0:-2]+rho[1:-1,2:]\
#        -4*rhon[1:-1,1:-1]+rhon[0:-2,1:-1]+rhon[2:,1:-1]+rhon[1:-1,0:-2]+rhon[1:-1,2:])/dx**2/2.#cells in the middle
       
#     Lap[1:-1,0]=epsilon**2*(-3*rho[1:-1,0]+rho[0:-2,0]+rho[2:,0]+rho[1:-1,1]\
#        -3*rhon[1:-1,0]+rhon[0:-2,0]+rhon[2:,0]+rhon[1:-1,1])/dx**2/2.#cells on the left line 
    
#     Lap[1:-1,-1]=epsilon**2*(-3*rho[1:-1,-1]+rho[0:-2,-1]+rho[2:,-1]+rho[1:-1,-2]\
#        -3*rhon[1:-1,-1]+rhon[0:-2,-1]+rhon[2:,-1]+rhon[1:-1,-2])/dx**2/2.#cells on the right line
    
#     Lap[0,1:-1]=epsilon**2*(-3*rho[0,1:-1]+rho[0,0:-2]+rho[0,2:]+rho[1,1:-1]\
#        -3*rhon[0,1:-1]+rhon[0,0:-2]+rhon[0,2:]+rhon[1,1:-1])/dx**2/2.#cells on the top line
    
#     Lap[-1,1:-1]=epsilon**2*(-3*rho[-1,1:-1]+rho[-1,0:-2]+rho[-1,2:]+rho[-2,1:-1]\
#        -3*rhon[-1,1:-1]+rhon[-1,0:-2]+rhon[-1,2:]+rhon[-2,1:-1])/dx**2/2.#cells on the bottom line
    
#     Lap[0,0]=epsilon**2*(-2*rho[0,0]+rho[1,0]+rho[0,1]\
#        -2*rhon[0,0]+rhon[1,0]+rhon[0,1])/dx**2/2.#cells on the top left corner
       
#     Lap[0,-1]=epsilon**2*(-2*rho[0,-1]+rho[1,-1]+rho[0,-2]\
#        -2*rhon[0,-1]+rhon[1,-1]+rhon[0,-2])/dx**2/2.#cells on the top right corner
       
#     Lap[-1,0]=epsilon**2*(-2*rho[-1,0]+rho[-2,0]+rho[-1,1]\
#        -2*rhon[-1,0]+rhon[-2,0]+rhon[-1,1])/dx**2/2.#cells on the bottom left corner
       
#     Lap[-1,-1]=epsilon**2*(-2*rho[-1,-1]+rho[-2,-1]+rho[-1,-2]\
#        -2*rhon[-1,-1]+rhon[-2,-1]+rhon[-1,-2])/dx**2/2.#cells on the bottom right corner
       
#     # Compute (n-1) u velocities
    
#     uhalf=-(Hc1[1:,:]-He1[1:,:]-Lap[1:,:]-Hc1[0:-1,:]+He1[0:-1,:]+Lap[0:-1,:])/dx
    
#     # Upwind u velocities
    
#     uhalfplus=np.zeros((n-1,n))
#     uhalfminus=np.zeros((n-1,n))
#     uhalfplus[uhalf > 0]=uhalf[uhalf > 0]
#     uhalfminus[uhalf < 0]=uhalf[uhalf < 0]
    
#     # Compute (n-1) v velocities
    
#     vhalf=-(Hc1[:,1:]-He1[:,1:]-Lap[:,1:]-Hc1[:,0:-1]+He1[:,0:-1]+Lap[:,0:-1])/dx
    
    
#     # Upwind u velocities
    
    
#     vhalfplus=np.zeros((n,n-1))
#     vhalfminus=np.zeros((n,n-1))
#     vhalfplus[vhalf > 0]=vhalf[vhalf > 0]
#     vhalfminus[vhalf < 0]=vhalf[vhalf < 0]
    

#     # Compute (n+1,n) x fluxes, including no-flux boundary conditions
    
#     Fxhalf=np.zeros((n+1,n))
    
#     # 1st order
#     Fxhalf[1:-1,:]=uhalfplus*mobility(rho[0:-1,:],choicemob)+uhalfminus*mobility(rho[1:,:],choicemob) 
    
#     # Compute (n+1) y fluxes, including no-flux boundary conditions
    
#     Fyhalf=np.zeros((n,n+1))
    
#     # 1st order
#     Fyhalf[:,1:-1]=vhalfplus*mobility(rho[:,0:-1],choicemob)+vhalfminus*mobility(rho[:,1:],choicemob) 
    
    
#     # initial state
    
#     #a)import and read the original image f
#     #img = np.load('input/dimage_0.npy')
#     #img = io.imread('d6.png', as_gray=True)*(-2)+1#initial density from given inpainting image
    
#     # rho0=np.reshape(rho0_1, (np.shape(rhon)))
#     rho0 = np.load('data/CH_image_initial.npy')
    
#     rho0 = np.reshape(rho0[0,:], (np.shape(rhon)))

   
#     #b)define lambda
#     def lam(lambda0): 
#      lar=np.full((28,28), lambda0)
#      #lar[:,11:17,:]=np.matrix((np.full((30,6,4),0)))
#      lar[5,:]=0
#      lar[13:15,:]=0
#      lar[20,:]=0
#      return np.reshape(lar, np.shape(rho))

    

#     E_i=rhon-rho+(Fxhalf[1:,:]-Fxhalf[:-1,:]+Fyhalf[:,1:]-Fyhalf[:,:-1])*dt/dx + lam(10000)*(rho0-rhon)*dt
#     #E_i=rhon-rho+(Fxhalf[1:,:]-Fxhalf[:-1,:]+Fyhalf[:,1:]-Fyhalf[:,:-1])*dt/dx + lam(0.2)*(rho0-rho)*dt
#     #E_i=rhon-rho+(Fxhalf[1:,:]-Fxhalf[:-1,:]+Fyhalf[:,1:]-Fyhalf[:,:-1])*dt/dx + lam(60)*(rho-rho0)*dt
   
#     return np.reshape(E_i,n*n)
 

