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
from GenerativeModel import LG, NLG
from smootherKalman import smoothKalman
import matplotlib.pyplot as plt
#%% Methods
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def returnNextBatch(batch_size, x, z):
    x_size = np.shape(x)[0]
    
    if x_size < batch_size:
        raise Exception("You are trying to get a batch that is larger than " +
                        "the amount of samples")
    
    idx = np.random.randint(x_size, size=batch_size)
    return x[idx], z[idx]

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

#nTimeSteps = 200
#nTrainSamples = 10000
#nTestSamples = 1#00#0
#zTrain is not used, but the method generateSamples returns it anyways.
#zTrain = np.zeros([nTrainSamples, nTimeSteps, dim])
#xTrain = np.zeros([nTrainSamples, nTimeSteps, dim])
#zTest = np.zeros([nTestSamples, nTimeSteps, dim])
#xTest = np.zeros([nTestSamples, nTimeSteps, dim])

# =============================================================================
# for i in range(0,nTrainSamples):
#     zTrain[i], xTrain[i] = model.generateSamples(nTimeSteps)
# =============================================================================
    
#for i in range(0,nTestSamples):
#    zTest[i], xTest[i] = model.generateSamples(nTimeSteps)
    
xTrain = np.load(r'C:/Project16/generateData/NLG200xTrainSin.dat')
zTrain = np.load(r'C:/Project16/generateData/NLG200zTrainSin.dat')
xTest = np.load(r'C:/Project16/generateData/NLG200xTestSin.dat')
zTest = np.load(r'C:/Project16/generateData/NLG200zTestSin.dat')
#xTest = xTest[0:1]
#zTest = zTest[0:1]
nTimeSteps = xTrain.shape[1]
nTrainSamples = xTrain.shape[0]

    
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
batch_size = 1000

nEps = 10 
observed = tf.placeholder(dtype = tf.float32, shape = [None, nTimeSteps, xDim])
hidden = tf.placeholder(dtype = tf.float32, shape = [None, nTimeSteps, zDim])
z0_tensor = tf.placeholder(dtype = tf.float32, shape = [None, zDim])


obs_series = tf.unstack(observed, axis=1)
hidden_series = tf.unstack(hidden, axis=1)

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

z_current = z0_tensor
for i in range(nTimeSteps):
#    h_comb.append(tf.nn.tanh(tf.matmul(z_current, W_comb) + b_comb) + h[0])
    h_comb.append(tf.nn.tanh(z_current*W_comb + b_comb) + h[i])

    #first get sigma i and mu i
    mu.append(tf.matmul(h_comb[i], W_mu) + b_mu)
    #sigma.append(tf.nn.softplus(tf.matmul(h_comb[i], W_sigma) + b_sigma))
    #get mean of lots of samples
    z_current = mu[i] #we need to sample this instead of just taking the mean (right?)
    
loss = tf.reduce_mean(tf.square(mu[0] - hidden_series[0])) 
for i in range (1, nTimeSteps):
    loss = loss + tf.reduce_mean(tf.square(mu[i] - hidden_series[i]))
    
loss_comp = tf.reduce_sum(tf.sqrt(1+tf.square(hidden - observed)) -1)
 
       
train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(loss) #doesnt converge with adam

#%%
'''
Kalman smoother for reference - smoothes test data
'''

x_kalmanShape = np.reshape(xTest[0], [nTimeSteps, xDim])
timeVec, kal_mu, kal_cov = smoothKalman(x_kalmanShape, model)
kal_mu = np.reshape(kal_mu, [nTimeSteps])
kal_stdev = np.sqrt(np.reshape(kal_cov, [nTimeSteps]))
    
#%% Training of the neural net
#with tf.Session() as sess:
#    init = tf.global_variables_initializer()
#    sess.run(init)
num_epochs = 2000
loss_array = np.array([])
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(num_epochs):
    #batch_obs, batch_hidden = returnNextBatch(batch_size, xTrain, zTrain)
    lossPlt = 0
    print('Epoch no. %d' % i)
    for j in range(np.int32(nTrainSamples/batch_size)):
        batch_obs = xTrain[j*batch_size:(j+1)*batch_size]
        batch_hidden = zTrain[j*batch_size:(j+1)*batch_size]
        #if i % 100 == 0:
        curr_loss = sess.run(loss, feed_dict={observed: batch_obs,
                                              hidden: batch_hidden,
                                              z0_tensor: np.reshape(z0, [1, 1])})
        lossPlt = lossPlt + curr_loss
            #print("step %d, loss: %g" % (i, curr_loss))
        train_step.run(feed_dict={observed: batch_obs,
                                                  hidden: batch_hidden,
                                                  z0_tensor: np.reshape(z0, [1, 1])})
    loss_array = np.append(loss_array, lossPlt*batch_size/nTrainSamples)
    
    _output = sess.run(mu, feed_dict={observed: xTest[0:1],
                                        z0_tensor: np.reshape(z0, [1, 1])})
    _output = np.array(_output).reshape(-1,nTimeSteps)
    
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1,2,1)
    plt.plot(loss_array)
    ax2 = fig.add_subplot(1,2,2)
    plt.scatter(np.arange(nTimeSteps),zTest[0],color='black')
#        plt.plot(np.arange(nTimeSteps),np.flip(np.reshape(_mean,(-1)),0),color='blue')
    plt.plot(np.arange(nTimeSteps),np.reshape(_output,(nTimeSteps)),color='blue')
#    plt.plot(np.arange(nTimeSteps),kal_mu,color='red')
    plt.show()
    print('loss: %g' % lossPlt)
Test_smoothed = sess.run(mu, feed_dict={observed: xTest,
                                        z0_tensor: np.reshape(z0, [1, 1])})
np.concatenate(Test_smoothed,axis=1).dump('NLGsinRNNsmoothed.dat')

#%% Save data
#np.array(Test_smoothed).reshape(-1,nTimeSteps).dump('NLNGSinRNNsmoothed.dat')
loss_array[0:1600].dump('NLGsinRNNmse.dat')
#plt.plot(loss_array)


#%% plots
CNN_smoothedLG = np.load(r'/home/hannes/Programming_projects/Project16/ConvNetSmoother/LNGRNNsmoothed.dat')
CNN_smoothedLG = CNN_smoothedLG[0:1]
#plt.plot(timeVec, kal_mu, color = 'blue', label = 'Kalman smoother')
#plt.fill_between(timeVec, kal_mu-kal_stdev, kal_mu+kal_stdev)
plt.scatter(timeVec, np.reshape(zTest[0:1], [nTimeSteps]), color = 'green', s = 5, label = 'True z')
#plt.plot(timeVec, np.reshape(Test_smoothed, [nTimeSteps]),'-', color = 'yellow', label = 'RNN smoothed'\
#         )
plt.plot(timeVec,CNN_smoothedLG.reshape(-1),color='red')
plt.legend()
plt.show()



