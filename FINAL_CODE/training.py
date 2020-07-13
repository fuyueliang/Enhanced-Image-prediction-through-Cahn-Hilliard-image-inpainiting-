# -*- coding: utf-8 -*-
"""

TRAINING: train the model for the subsequent prediction

Created on Thu Aud 01 2019
@author: FL18
"""
import numpy as np 
import matplotlib.pyplot as plt
# for model consrtuction 
import tensorflow as tf
import mnist
from keras.models import Sequential #the most common type of model is a stack of layers: the tf.keras.Sequential model
from keras.layers import Dense
from keras.utils import to_categorical
# for table construction 
from prettytable import PrettyTable
from prettytable import from_db_cursor
# import data from test image
import mnist
from PIL import Image
# for import python files 
import os
dirpath = os.getcwd()
import sys
sys.path.append(dirpath+'\\functions')
from initial_conditions_1 import initial_conditions_1




# initial_conditions
choiceinitialcond=1 # Select the configuration of the problem
num, n, dx, x, epsilon, dt, tmax, ntimes, pot, theta, thetac, choicemob = initial_conditions_1(choiceinitialcond) 

###############################
# BUILD MODEL
###############################
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()


# Normalize the test image   CH Equation work for (-1,1)
train_images = (train_images / 255) *2-1
test_images = (test_images / 255) *2-1

# Flatten the test images
train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))

# Build the model
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)), 
  Dense(64, activation='relu'),  
  Dense(10, activation='softmax'), 
])



# Compile the model
model.compile(
  optimizer='adam',    
  loss='categorical_crossentropy', 
  metrics=['accuracy'], 
)

# Train the model
model.fit( 
  train_images,
  to_categorical(train_labels),
  epochs=5, 
  batch_size=32,  
)

# Evaluate the model
model.evaluate(
 test_images,
 to_categorical(test_labels)
)

# Save the model
model.save_weights('model.h5')
