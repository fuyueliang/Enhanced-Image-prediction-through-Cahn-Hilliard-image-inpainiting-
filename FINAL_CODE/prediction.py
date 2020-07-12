# -*- coding: utf-8 -*-
"""
Created on Thu Aud 01 2019

@author: FL18
"""
import numpy as np 
import matplotlib.pyplot as plt
# for model consrtuction 
import tensorflow as tf
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
# INPUT 
###############################

test_labels = mnist.test_labels()


# load the model
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

model.load_weights('model.h5')




#####################
# # Prediction
# ####################

# input predicting image

img_p = np.load('data/CH_image_final.npy')
img_p = np.reshape(img_p, (-1,784))
img_pi = np.load('data/CH_image_initial.npy')
img_pi = np.reshape(img_pi, (-1,784))

    
# Prediction

predictions = model.predict(img_p[:num])
preds = np.argmax(predictions, axis=1)
predictions_i = model.predict(img_pi[:num])
preds_i = np.argmax(predictions_i, axis=1)
# Print our model's predictions.
#print("predicted probabilities: ", predictions)




########################
# TABLE OF PREDICTION
########################
#Print the result and the accuracy as a table
# Header
table = PrettyTable(['Image', 'True', 'Predictions_i','prediction_f'])
# column_name = (['Image', 'True', 'Predictions','Match'])

for x in range(num):    
    # if test_labels[x] == preds[x]:
    #       table.add_column(column_name[3], 'True')
    # else:
    #       table.add_column(column_name[3], 'False')
    table.add_row([str(x), test_labels[x], preds_i[x], preds[x]])
print(table)
try:
  np.save('data/table.npy', np.column_stack((test_labels[0:num], preds_i[0:num], preds[0:num])))
except:
  print("meh")




#############################
# Accuracy of the prediction
#############################
N_CORRECT = 0
N_ITEMS_SEEN = 0


def reset_running_variables():
    """ Resets the previous values of running variables to zero """
    global N_CORRECT, N_ITEMS_SEEN, N_CORRECT_i
    N_CORRECT = 0
    N_CORRECT_i = 0
    N_ITEMS_SEEN = 0
    # N_ITEMS_SEEN_i = 0


def update_running_variables(labs, preds, preds_i):
    global N_CORRECT, N_ITEMS_SEEN, N_CORRECT_i
    N_CORRECT += (labs == preds).sum()
    N_CORRECT_i += (labs == preds_i).sum()
    N_ITEMS_SEEN += labs.size
    

def calculate_accuracy():
    global N_CORRECT, N_ITEMS_SEEN, N_CORRECT_i
    return float(N_CORRECT_i) / N_ITEMS_SEEN, float(N_CORRECT) / N_ITEMS_SEEN

# Apply the functions defined above
reset_running_variables()

for y in range(num):

    update_running_variables(labs = test_labels[y], preds_i = preds_i[y], preds = preds[y])

accuracy = calculate_accuracy()          
print ('SCORE(initial,final):', accuracy)


