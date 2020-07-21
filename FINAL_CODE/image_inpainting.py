# -*- coding: utf-8 -*-
"""
IMAGE INPAINTING: restore binary images using Cahn-Hilliard equation

Created on Thu Aud 01 2019
@author: FL18
"""

###########################
# VARIABLES can be changed
##########################
# run the 2D file: secondstep: time, (epsilon)
# initial_condition_1: num,tmax,(epsilon) 
# lambda = 10000, epsilon_1 = 1.8, epsilon_2 = 0.6

##########################
# IMPORT PACKAGES
##########################

import numpy as np
# for import image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from skimage import io
# for import python files 
import os
dirpath = os.getcwd()
import sys
sys.path.append(dirpath+'\\functions')
from initial_conditions_1 import initial_conditions_1
from Euler_implicit_1 import Hc1_con
from Euler_implicit_1 import Hc2_con
from Euler_implicit_1 import He1_exp
from Euler_implicit_1 import mobility
# import data from test image
import mnist
from PIL import Image
# for first step of the Cahn-Hilliard equation
from skimage import io
from scipy import optimize
# from time import process_time
import time
t_start = time.clock()
# for model consrtuction 
import tensorflow as tf
from keras.models import Sequential #the most common type of model is a stack of layers: the tf.keras.Sequential model
from keras.layers import Dense
from keras.utils import to_categorical
# for table construction 
from prettytable import PrettyTable
from prettytable import from_db_cursor
# for data storage
import sqlite3 as sql



plt.rc('text',usetex=False)
plt.rc('font',family='serif')

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()


# Normalize the test image   CH Equation works for (-1,1)
train_images = (train_images / 255) *2-1
test_images = (test_images / 255) *2-1

# Flatten the test images
train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))


##############################
# Cahn-Hilliard EUQTION 
##############################


# initial_conditions

choiceinitialcond=1 # Select the configuration of the problem
num, n, dx, x, epsilon, dt, tmax, ntimes, pot, theta, thetac, choicemob = initial_conditions_1(choiceinitialcond) 
CH_image_final = np.zeros([num,n*n])
CH_image_initial = np.zeros([num,n*n]) #image matrix


for i in range(num):
    
    t=np.zeros(ntimes+1) # Time vector
    rho=np.zeros([n*n,ntimes+1]) # Density matrix

    # input data
    img = test_images[i,:]
        
    
    

    # create damage and save it for the inpainting later
    # either Random pixels or Random rows, they are just different ways of creating damage
    ###################
    # RANDOM PIXELS
    ###################
    # define lambda function used for inpainting
    lam=np.full((n*n), 9000)
    indices = np.random.choice(np.arange(img.size), replace=False, size=int(img.size * 0.96))
    img[indices]=0
    lam[indices]=0 

    ###################
    # RANDOM ROWS
    ###################
    # define lambda function used for inpainting
    lam=np.full((n,n), 1000)
    img = np.reshape(img,(n,n))
    c = [np.random.choice(img.shape[0], size = 26, replace=False)]
    img[c,:]=0
    lam[c,:]=0

    # reshape the one-line vector to a 28X28 matrix
    img = np.reshape(img,(n,n))
    lam = np.reshape(lam,(n,n))
    

    # data save and figures
    np.save('input/dimage_'+str(i), img) 


    #input data
    img = np.load('input/dimage_'+str(i)+'.npy')
    rho0_1 = np.reshape(img, n*n)
    rho[:,0]=rho0_1 ##### rho0_1.shape = (784,) # First column of density matrix is initial density
    CH_image_initial[i,:] = rho[:,0]
        
    ########################
    # DEFINE FUNCTION: FLUX 
    #######################


    def Euler_implicit_1(rho,rhon,n,dx,dt,epsilon,ntimes,pot,theta,thetac,choicemob):
        
        # Create matrix rho
        
        rho=np.reshape(rho,(n,n))
        rhon=np.reshape(rhon,(n,n)) # rho in the future rho.shape = (28,28)
        
        # Define variation of free energy
        
        # a) Hc: contractive (convex) part of the free energy (treated implicitly))
        # Hc1 is the first derivative of Hc
        
        Hc1= Hc1_con(rhon,pot,theta)
        
        # b) He: expansive (concave) part of the free energy (treated explicitly)
        # He1 is the first derivative of He    
        
        He1= He1_exp(rho,pot,thetac)
        
        # c) Laplacian (treated semi-implicitly)
        #see the difference for cells in the middle, at the corner, lying on the line

        Lap=np.zeros((n,n))
        
        Lap[1:-1,1:-1]=epsilon**2*(-4*rho[1:-1,1:-1]+rho[0:-2,1:-1]+rho[2:,1:-1]+rho[1:-1,0:-2]+rho[1:-1,2:]\
        -4*rhon[1:-1,1:-1]+rhon[0:-2,1:-1]+rhon[2:,1:-1]+rhon[1:-1,0:-2]+rhon[1:-1,2:])/dx**2/2.#cells in the middle
        
        Lap[1:-1,0]=epsilon**2*(-3*rho[1:-1,0]+rho[0:-2,0]+rho[2:,0]+rho[1:-1,1]\
        -3*rhon[1:-1,0]+rhon[0:-2,0]+rhon[2:,0]+rhon[1:-1,1])/dx**2/2.#cells on the left line 
        
        Lap[1:-1,-1]=epsilon**2*(-3*rho[1:-1,-1]+rho[0:-2,-1]+rho[2:,-1]+rho[1:-1,-2]\
        -3*rhon[1:-1,-1]+rhon[0:-2,-1]+rhon[2:,-1]+rhon[1:-1,-2])/dx**2/2.#cells on the right line
        
        Lap[0,1:-1]=epsilon**2*(-3*rho[0,1:-1]+rho[0,0:-2]+rho[0,2:]+rho[1,1:-1]\
        -3*rhon[0,1:-1]+rhon[0,0:-2]+rhon[0,2:]+rhon[1,1:-1])/dx**2/2.#cells on the top line
        
        Lap[-1,1:-1]=epsilon**2*(-3*rho[-1,1:-1]+rho[-1,0:-2]+rho[-1,2:]+rho[-2,1:-1]\
        -3*rhon[-1,1:-1]+rhon[-1,0:-2]+rhon[-1,2:]+rhon[-2,1:-1])/dx**2/2.#cells on the bottom line
        
        Lap[0,0]=epsilon**2*(-2*rho[0,0]+rho[1,0]+rho[0,1]\
        -2*rhon[0,0]+rhon[1,0]+rhon[0,1])/dx**2/2.#cells on the top left corner
        
        Lap[0,-1]=epsilon**2*(-2*rho[0,-1]+rho[1,-1]+rho[0,-2]\
        -2*rhon[0,-1]+rhon[1,-1]+rhon[0,-2])/dx**2/2.#cells on the top right corner
        
        Lap[-1,0]=epsilon**2*(-2*rho[-1,0]+rho[-2,0]+rho[-1,1]\
        -2*rhon[-1,0]+rhon[-2,0]+rhon[-1,1])/dx**2/2.#cells on the bottom left corner
        
        Lap[-1,-1]=epsilon**2*(-2*rho[-1,-1]+rho[-2,-1]+rho[-1,-2]\
        -2*rhon[-1,-1]+rhon[-2,-1]+rhon[-1,-2])/dx**2/2.#cells on the bottom right corner
        
        # Compute (n-1) u velocities
        
        uhalf=-(Hc1[1:,:]-He1[1:,:]-Lap[1:,:]-Hc1[0:-1,:]+He1[0:-1,:]+Lap[0:-1,:])/dx
        
        # Upwind u velocities
        
        uhalfplus=np.zeros((n-1,n))
        uhalfminus=np.zeros((n-1,n))
        uhalfplus[uhalf > 0]=uhalf[uhalf > 0]
        uhalfminus[uhalf < 0]=uhalf[uhalf < 0]
        
        # Compute (n-1) v velocities
        
        vhalf=-(Hc1[:,1:]-He1[:,1:]-Lap[:,1:]-Hc1[:,0:-1]+He1[:,0:-1]+Lap[:,0:-1])/dx
        
        
        # Upwind u velocities
        
        
        vhalfplus=np.zeros((n,n-1))
        vhalfminus=np.zeros((n,n-1))
        vhalfplus[vhalf > 0]=vhalf[vhalf > 0]
        vhalfminus[vhalf < 0]=vhalf[vhalf < 0]
        

        # Compute (n+1,n) x fluxes, including no-flux boundary conditions
        
        Fxhalf=np.zeros((n+1,n))
        
        # 1st order
        Fxhalf[1:-1,:]=uhalfplus*mobility(rho[0:-1,:],choicemob)+uhalfminus*mobility(rho[1:,:],choicemob) 
        
        # Compute (n+1) y fluxes, including no-flux boundary conditions
        
        Fyhalf=np.zeros((n,n+1))
        
        # 1st order
        Fyhalf[:,1:-1]=vhalfplus*mobility(rho[:,0:-1],choicemob)+vhalfminus*mobility(rho[:,1:],choicemob) 
        
        
        # initial state
        
        rho0 = np.reshape(rho0_1, (np.shape(rhon)))

    
        #b)define lambda

        E_i=rhon-rho+(Fxhalf[1:,:]-Fxhalf[:-1,:]+Fyhalf[:,1:]-Fyhalf[:,:-1])*dt/dx + lam*(rho0-rhon)*dt
    
        return np.reshape(E_i,n*n) #E_i.shape = (784,)



    ##########################
    # run the 2D file
    ##########################
    def main():
        global n, x, dx, dt,epsilon,ntimes,pot,theta,thetac,choicemob          

        # tic = time.clock()

        for ti in np.arange(ntimes):

            # Euler implicit
                
            rho[:,ti+1],infodict, ier, mesg = optimize.fsolve(lambda rhon: Euler_implicit_1(rho[:,ti],rhon,n,dx,dt,epsilon,ntimes,pot,theta,thetac,choicemob), rho[:,ti], full_output = True)
            

            t[ti+1]=t[ti]+dt
            
            print('--------------------')
            print('Time: ',t[ti])
            print(['L1 norm of the difference between the new and old state: ',np.linalg.norm(rho[:,ti+1]-rho[:,ti],1)])

            if np.linalg.norm(rho[:,ti+1]-rho[:,ti],1) <  0.0001 :
                break

            if np.linalg.norm(rho[:,ti+1]-rho[:,ti],1) > 1000:
                break

            # Second step: TWO-STEP method---sharp the edges of the images while the previous step is to execute a topological reconnection of the shape with diffused edges.

            if t[ti] > 1:   
                epsilon = 0.5
        
        # save data
        np.save('data/rho_'+str(i), rho)
            
        CH_image_final[i,:] = rho[:,-1]
        
        # plot thr figures

        figure1 = plt.figure(figsize=(10,6.5))
        #plt.title(r'Final $\phi$ ', fontsize=36)
        plt.imshow(np.reshape(rho[:,0],(n,n)), cmap='Greys', vmin=-1, vmax=1,aspect='auto',extent=[x[0],x[-1],x[0],x[-1]])
        #plt.ylabel(r'$y$', fontsize=25)
        plt.xlabel(r'$x$', fontsize=25)
        plt.colorbar()
        plt.show()
        plt.close()
        figure1.savefig('figures/INITIAL_'+str(i)+'.png', bbox_inches='tight')

        figure2 = plt.figure(figsize=(10,6.5))
        #plt.title(r'Final $\phi$ ', fontsize=36)
        plt.imshow(np.reshape(rho[:,-1],(n,n)), cmap='Greys', vmin=-1, vmax=1,aspect='auto',extent=[x[0],x[-1],x[0],x[-1]])
        #plt.ylabel(r'$y$', fontsize=25)
        plt.xlabel(r'$x$', fontsize=25)
        plt.colorbar()
        plt.show()
        plt.close()
        figure2.savefig('figures/Inpaintings_'+str(i)+'.png', bbox_inches='tight')


    if __name__ == '__main__':
        main()        


    # save data
    np.save('data/CH_image_final', CH_image_final)
    np.save('data/CH_image_initial', CH_image_initial)
    print ('-----------------------------Inpainting of image '+str(i+1)+' completed------------------------------------')



os.system('python prediction.py')
    
