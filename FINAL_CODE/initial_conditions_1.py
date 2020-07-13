# -*- coding: utf-8 -*-
"""
INITIAL_CONDITION: the initial conditions for numerical computation

Created on Mon Apr 29 12:58:49 2019
@author: sp3215
"""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.integrate import quad

##################
# DEFINE FUNCTION: INITIAL CONDITIONS
##################
    
def initial_conditions_1(choice):
    
    if choice==1: #Random initial configuration
        num=100
        n=28 # Number of cells per row
        dx=1
        x= np.linspace(-n*dx/2,n*dx/2,n)
        #rho0=0.5*np.random.random_sample([n*n])-0.25 #Initial density
        #img = io.imread('d6.png', as_gray=True)*(-2)+1#initial density from given inpainting image
        #img = np.load('dimage.npy')
        #rho0=np.reshape(img, n*n)
        epsilons = 1.5 # Parameter epsilon   
        #epsilons = 2.2
        dt=0.1 # Time step
        tmax=6
        ntimes=int(tmax/dt)# 4000 # Number of time steps        
        pot=1 # Choice of the potential: 1 is double-well, 2 is logarithmic
        theta=0.2 # Absolut temperature for the logarithmic potential
        thetac=1. # Critical temperature for the logarithmic potential
        choicemob=2 # Choice of mobility. 1 is constant mobility, 2 is 1-rho**2
        

        #save initial condition
        #np.save('rho0', rho0)
        
        
        
    
    
    return num, n, dx, x, epsilons, dt, tmax, ntimes, pot, theta, thetac, choicemob

        
