#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:37:58 2017

@author: hannes
"""
import numpy as np
import tensorflow as tf
from GenerativeModel import LG
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
model = LG(zDim, xDim, A, B, Q, R, z0, Q0)

nTimeSteps = 50
nTrainSamples = 1#000#0
nTestSamples = 1#00#0
zTrain = np.zeros([nTrainSamples, nTimeSteps, dim])
xTrain = np.zeros([nTrainSamples, nTimeSteps, dim])
zTest = np.zeros([nTestSamples, nTimeSteps, dim])
xTest = np.zeros([nTestSamples, nTimeSteps, dim])

for i in range(0,nTrainSamples):
    zTrain[i], xTrain[i] = model.generateSamples(nTimeSteps)
    
for i in range(0,nTestSamples):
    zTest[i], xTest[i] = model.generateSamples(nTimeSteps)
'''
Neural net part
'''
#%%
x_dim = dim
z_dim = dim
batch_size = 1
#input to our model
obs_placeholder = tf.placeholder(tf.float32, shape = (nTimeSteps, x_dim)) 
#output, this is what we want to predict
latent_placeholder = tf.placeholder(tf.float32, shape = (nTimeSteps, z_dim)) 

#layer 1 atm and not convolution, for convolution I should add filter size
W = tf.Variable(tf.zeros([x_dim, z_dim]), dtype=tf.float32)
b = tf.Variable(tf.zeros(z_dim), dtype=tf.float32)

y = tf.nn.softmax(tf.matmul(obs_placeholder, W) + b)

#loss function
loss = tf.reduce_mean(tf.squared_difference(y, latent_placeholder))

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#function for initializing variables
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train, {obs_placeholder: xTrain, y: zTrain})
    
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {obs_placeholder: xTrain[0], 
                                     y: latent_placeholder[0]})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


