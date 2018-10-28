#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:37:58 2017

@author: hannes
"""
import numpy as np
import tensorflow as tf
from GenerativeModel import NLG
'''
Methods
'''
#%%
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_layer(input, num_input_channels, 
              filter_size, num_filters, use_pooling=True):
    '''1d'''
    shape = [filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    
    
    

'''
Input parameters
(zDim, xDim, A, B, Q, R, z0, Q0)
'''
#%%
dim = 1
zDim = dim
xDim = dim
z0 = np.zeros(dim)
A = 0.9*np.identity(dim)
B = A
Q = np.identity(dim)
R = Q
Q0 = Q


'''generate data'''
#%%

model = NLG(zDim, xDim, A, B, Q, R, z0, Q0)

nTimeSteps = 50
<<<<<<< HEAD
nTrainSamples = 1000#0
nTestSamples = 1#00#0
zTrain = np.zeros([nTrainSamples, nTimeSteps, dim])
xTrain = np.zeros([nTrainSamples, nTimeSteps, dim])
zTest = np.zeros([nTestSamples, nTimeSteps, dim])
xTest = np.zeros([nTestSamples, nTimeSteps, dim])

for i in range(0,nTrainSamples):
    zTrain[i], xTrain[i] = model.generateSamples(nTimeSteps)
    
for i in range(0,nTestSamples):
    zTest[i], xTest[i] = model.generateSamples(nTimeSteps)

#Reshape for 1D problem    
zTrain = zTrain.reshape([nTrainSamples, nTimeSteps])
xTrain = xTrain.reshape([nTrainSamples, nTimeSteps])
zTest = zTest.reshape([nTestSamples, nTimeSteps])
xTest = xTest.reshape([nTestSamples, nTimeSteps])

'''
Neural net part
'''
#%%

x_dim = dim
z_dim = dim
batch_size = 2
sess = tf.InteractiveSession()
#input to our model
#obs_placeholder = tf.placeholder(tf.float32, shape = (None, nTimeSteps, x_dim))
obs_placeholder = tf.placeholder(tf.float32,shape=(None,nTimeSteps)) 
#output, this is what we want to predict
#latent_placeholder = tf.placeholder(tf.float32, shape = (None, nTimeSteps, z_dim))
latent_placeholder = tf.placeholder(tf.float32,shape=(None,nTimeSteps)) 


#layer 1 atm and not convolution, for convolution I should add filter size
W = tf.Variable(tf.zeros([nTimeSteps, nTimeSteps]), dtype=tf.float32)
b = tf.Variable(tf.zeros([nTimeSteps]), dtype=tf.float32)

#y = tf.nn.softmax(tf.matmul(obs_placeholder, W) + b)
y = tf.matmul(obs_placeholder, W) + b

#loss function
loss = tf.reduce_mean(tf.squared_difference(y, latent_placeholder))

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#function for initializing variables
init = tf.global_variables_initializer()

#sess = tf.Session()
sess.run(init)
feed = {obs_placeholder: xTrain, latent_placeholder: zTrain}


for i in range(2000):
    sess.run(train, feed)
    if i%100 == 0:
        print("hej")
        print(sess.run(loss, feed))
    

print("Finished training")
print()

feedTest = {obs_placeholder:xTest,latent_placeholder:zTest}
print(sess.run(loss,feedTest))

