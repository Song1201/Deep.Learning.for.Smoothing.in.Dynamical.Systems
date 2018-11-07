#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:49:14 2017

@author: hannes
tänk poå att vi har en flervariabel gaussfördelning med medelvärde 
(mu1, mu2, ..., muT) och bara diagonalgrejer i kovariansen (dvs bara varians)
efter att vi har implementerat det kan vi börja fundera på kovarianser
"""
import numpy as np
import tensorflow as tf
from GenerativeModel import LG
from smootherKalman import smoothKalman
import matplotlib.pyplot as plt
#%% Methods
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)

def returnNextBatch(batch_size, nTimeSteps):
    eps = np.random.normal(0,1,[batch_size*nTimeSteps])
    eps = np.reshape(eps, [batch_size*nTimeSteps, 1])
    return eps

def getELBO(x, mu, sigma, eps_series, model, batch_size, nTimeSteps):
    zDim, xDim, A, B, Q, R, z0, Q0 = model.returnParameters()
    elbo = 0
    for sample in range(batch_size):
        z = []
        for i in range(nTimeSteps):
            eps_id = sample*nTimeSteps+i
            sgm = sigma[i]
            z.append(mu[i]+eps_series[(eps_id)]*sgm)
            log_cond = - 0.5*tf.square((x[i])-B*z[-1])/(R*R)
         
            N0 = tf.distributions.Normal(mu[i], sgm)
            if i == 0:
                N1 = tf.distributions.Normal(np.float64(0), Q, name='N1_0')
            else:
                N1 = tf.distributions.Normal(A*z[-2], Q)

            kl = tf.contrib.distributions.kl_divergence(N0, N1) 
            
            elbo = elbo + log_cond - kl

    return elbo/batch_size    

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
B = 3.5*np.identity(dim)
Q = np.identity(dim)
R = Q
Q0 = Q


'''generate data'''
#%%

model = LG(zDim, xDim, A, B, Q, R, z0, Q0)

nTimeSteps = 200
nTrainSamples = 10000
nTestSamples = 1#00#0

# =============================================================================
# zTest = np.zeros([nTestSamples, nTimeSteps, dim])
# xTest = np.zeros([nTestSamples, nTimeSteps, dim])
#     
# for i in range(0,nTestSamples):
#     zTest[i], xTest[i] = model.generateSamples(nTimeSteps)
# =============================================================================
    

#xTest = np.load('/home/hannes/Programming_projects/Project16/generateData/LG200xTest.dat')
#zTest = np.load('/home/hannes/Programming_projects/Project16/generateData/LG200zTest.dat')

xTest = np.load(r'C:\Project16\generateData\LG200xTest.dat')
zTest = np.load(r'C:\Project16\generateData\LG200zTest.dat')

#zTry = np.load('/home/hannes/Programming_projects/Project16/generateData/LG200xTest.dat')
#xTry = np.load('/home/hannes/Programming_projects/Project16/generateData/LG200zTest.dat')


zTry = np.load('/home/hannes/Programming_projects/Project16/generateData/LG200xTrain.dat')
xTry = np.load('/home/hannes/Programming_projects/Project16/generateData/LG200zTrain.dat')


#xTry = xTry[0]
#zTry = zTry[0]
xTry = np.reshape(xTry, [-1, 200, 1])

xTest = xTest[0]
zTest = zTest[0]

xTest = np.reshape(xTest, [1, 200, 1])
 
    
#Reshape for 1D problem    
# =============================================================================
# zTrain = zTrain.reshape([nTrainSamples, nTimeSteps])
# xTrain = xTrain.reshape([nTrainSamples, nTimeSteps])
# =============================================================================

'''
Neural net part
'''
#%%
'''
nEps: number of samples from the approximate posterior, has same role as batch 
size has for regular stochastic gradient descent
'''
batch_size = 10
#sess = tf.InteractiveSession()
#sess.run(tf.global_variables_initializer())
nEps = 10 
observed = tf.placeholder(dtype = tf.float64, shape = [None, nTimeSteps, xDim])
eps_sampled = tf.placeholder(dtype = tf.float64, shape = [nTimeSteps*batch_size, zDim])
z0_tensor = tf.placeholder(dtype = tf.float64, shape = [None, zDim])


obs_series = tf.unstack(observed, axis=1)
eps_series = tf.unstack(eps_sampled, axis=0)

#forward pass

W = weight_variable([xDim, xDim])
b = bias_variable([xDim])
W_last = weight_variable([xDim, 1])
W_comb = tf.concat([W, W_last], axis=0) #try axis = 1 if I get a weird error

h = []

h.append(tf.nn.relu(tf.matmul(obs_series[-1], W) + b))

for i in range(2,nTimeSteps+1):
    ipt = tf.concat([h[-1], obs_series[-i]], axis = 1)
    h.append(tf.nn.relu(tf.matmul(ipt, W_comb)+b))
    
h.reverse()   

W_comb = weight_variable([zDim, zDim])
b_comb = bias_variable([zDim])

W_mu = weight_variable([zDim, zDim])
b_mu = bias_variable([zDim])

W_sigma = weight_variable([zDim, zDim])
b_sigma = bias_variable([zDim])

h_comb = []
mu = []
sigma = []
#%%
z_current = z0_tensor
for i in range(nTimeSteps):
#    h_comb.append(tf.nn.tanh(tf.matmul(z_current, W_comb) + b_comb) + h[0])
    h_comb.append(tf.nn.tanh(z_current*W_comb + b_comb) + h[i])

    #first get sigma i and mu i
    mu.append(tf.matmul(h_comb[i], W_mu) + b_mu)
    sigma.append(tf.nn.softplus(tf.matmul(h_comb[i], W_sigma) + b_sigma))
    #get mean of lots of samples
    z_current = mu[i] #we need to sample this instead of just taking the mean (right?)

loss = getELBO(obs_series, mu, sigma, eps_series, model, batch_size, nTimeSteps)
loss = tf.reduce_mean(loss)
    
#train_step = tf.train.AdamOptimizer(1e-4).minimize(-loss) 
opt = tf.train.AdamOptimizer(1e-4)
train_step = opt.minimize(-loss)
#grads_and_vars = opt.compute_gradients(-loss)
#train_step = opt.apply_gradients(grads_and_vars)
    
#%% Training of the neural net
#with tf.Session() as sess:
#    init = tf.global_variables_initializer()
#    sess.run(init)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
<<<<<<< HEAD
for i in range(200000):
    batch_eps = returnNextBatch(batch_size, nTimeSteps)
    index = np.random.randint(0,20)
    xCurr = xTry[index:index+1]
    zCurr = xTry[index:index+1]
#    curr_loss = sess.run(loss, feed_dict={observed: xTest, eps_sampled: batch_eps,
#                                          z0_tensor: np.reshape(z0, [1, 1])})
    if i % 100 == 0:
        curr_loss = sess.run(loss, feed_dict={observed: xTest, eps_sampled: 
            batch_eps, z0_tensor: np.reshape(z0, [1, 1])})
        print("step %d, ELBO: %g" % (allElbo.shape[0], curr_loss))
        curr_mu = sess.run(mu, feed_dict={observed: xTest, 
                                          z0_tensor: np.reshape(z0, [1, 1])})
        curr_sigma = sess.run(sigma, feed_dict={observed: xTest, 
                                          z0_tensor: np.reshape(z0, [1, 1])})
        curr_mu = np.reshape(curr_mu, [nTimeSteps])
        curr_sigma = np.reshape(curr_sigma, [nTimeSteps])
        x_kalmanShape = np.reshape(xTest, [nTimeSteps, xDim])
        timeVec, kalmanSmoothed, kal_var = smoothKalman(x_kalmanShape, model)
        fig = plt.figure(figsize=(14,5))
        ax1 = fig.add_subplot(1,2,1)
        plt.plot(timeVec, kalmanSmoothed, color = 'red', label = 'Kalman smoother')
        kal_stdev = np.sqrt(kal_var)
        plt.fill_between(timeVec, np.reshape(kalmanSmoothed-kal_stdev,[nTimeSteps]), 
                         np.reshape(kalmanSmoothed+kal_stdev, [nTimeSteps]), color = 'xkcd:light red')
        plt.plot(timeVec, curr_mu, label = 'BBVI', color = 'blue')
        plt.fill_between(timeVec, curr_mu-curr_sigma, curr_mu+curr_sigma, color = 'xkcd:grey blue')
        plt.scatter(timeVec, np.reshape(zTest, [nTimeSteps]), color = 'green', s = 5, label = 'True z')
        plt.legend()
        
        ax2 = fig.add_subplot(1,2,2)
        plt.plot(allElbo)
        plt.show()
    train_step.run(feed_dict={observed: xCurr,
                                              eps_sampled: batch_eps,
                                              z0_tensor: np.reshape(z0, [1, 1])})
#%%    
mu_series = sess.run(mu, feed_dict={observed: xTest, 
                                        z0_tensor: np.reshape(z0, [1, 1])})
sigma_series = sess.run(sigma, feed_dict={observed: xTest, 
                                        z0_tensor: np.reshape(z0, [1, 1])})
    

#%%
'''
Kalman smoother for reference - smoothes test data
'''
x_kalmanShape = np.reshape(xTest, [nTimeSteps, xDim]) #ska stå xTest sen 
timeVec, kal_mu, kal_stdev = smoothKalman(x_kalmanShape, model)
kal_mu = np.reshape(kal_mu, [nTimeSteps])
kal_stdev = np.reshape(kal_stdev, [nTimeSteps])
#%%
#mu_series = [-val for val in mu_series]
mu_plot = np.reshape(mu_series, [nTimeSteps])
sigma = np.reshape(sigma_series, [nTimeSteps])

#%% plots
plt.plot(timeVec, kal_mu, color = 'red', label = 'Kalman smoother')
plt.scatter(timeVec, np.reshape(zTry[1:2], [nTimeSteps]), color = 'green', s = 5, label = 'True z')
plt.plot(timeVec, mu_plot, color = 'blue', label = 'BBVI')
#plt.fill_between(timeVec, mu-sigma, mu+sigma, color = 'xkcd:light red', label = 'stdev BBVI')
#plt.scatter(timeVec, np.reshape(xTest, [nTimeSteps]), color = 'black', s = 2)
plt.legend()
