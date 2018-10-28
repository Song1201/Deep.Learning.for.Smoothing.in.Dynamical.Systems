from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from smootherKalman import smoothKalman
from GenerativeModel import LG

#%% Import and preprocess data
# x should be reversed and then send into the network!!!!!!
# x should be reversed and then send into the network!!!!!!
# x should be reversed and then send into the network!!!!!!
xTrain = np.load('xTrainData.dat')[0:5000,0:25]
zTrain = np.load('zTrainData.dat')[0:5000,0:25]
xTest = np.load('xTestData.dat')[0:5000,0:25]
zTest = np.load('zTestData.dat')[0:5000,0:25]


# defaulted x,z has the same number of time steps
nTimeSteps = xTrain.shape[1]
# Only use one-dimensional data for the reason of simpicity and the article also only used 
# one-dimensional.
xDim = xTrain.shape[2]
zDim = zTrain.shape[2]
# reverse the order of data to implement backward RNN

# Set the parameters for generative model
A = 0.9*np.identity(xDim)
B = -A
Q = np.identity(xDim)
R = Q
Q0 = Q

model = LG(zDim, xDim, A, B, Q, R, 0, Q0)

#%% Set training parameters.
num_epochs = 100
stateSize = 4 # can be dicided arbitrarily
batchSize = 50
num_batches = np.int64(xTrain.shape[0]/batchSize)

#%% Build the computational graph
xPlaceholder = tf.placeholder(tf.float32, [nTimeSteps,batchSize, xDim])

initState = tf.placeholder(tf.float32, [batchSize, stateSize])
# we want to output a number, so output size is 1
W = tf.Variable(np.random.rand(nTimeSteps,stateSize,1),dtype=tf.float32)
b = tf.Variable(np.zeros((nTimeSteps,1,1)), dtype=tf.float32)


cell = tf.nn.rnn_cell.BasicRNNCell(stateSize,reuse=None)

# Forward pass

# set time_major=True to speed up the computation, but need to transpose the input and output
# to be [nTimeSteps,batchSize,dim]

stateSeries,finalState = tf.nn.dynamic_rnn(cell,xPlaceholder,initial_state=initState,
                                                time_major=True) 
#%%   
hRight = tf.matmul(stateSeries,W)+b
hRight = tf.reverse(hRight,axis=[0])                                        

z0 = tf.placeholder(tf.float32,[1])
Wz = tf.Variable(np.random.rand(1),dtype=tf.float32)
bz = tf.Variable(np.zeros((1)),dtype=tf.float32)
Wmu = tf.Variable(np.random.rand(1),dtype=tf.float32)
bmu = tf.Variable(np.zeros((1)),dtype=tf.float32)
Wvar = tf.Variable(np.random.rand(1),dtype=tf.float32)
bvar = tf.Variable(np.zeros((1)),dtype=tf.float32)
mean = tf.zeros((1,batchSize,xDim))
var = tf.zeros((1,batchSize,xDim)) # variance
#zSampled = tf.zeros((batchSize,xDim))
zPrevious = tf.reshape(tf.tile(z0,[batchSize]),(batchSize,xDim))
#hCombined = []

for t in range(nTimeSteps):
#    hCombined.append(0.5*(tf.tanh(Wz*zPrevious+bz)+hRight[t]))
    hCombined = 0.5*(tf.tanh(Wz*zPrevious+bz)+hRight[t])
#    meant = Wmu*hCombined[-1] + bmu
    meant = Wmu*hCombined + bmu
#    mean = tf.concat((mean,meant),1)
    mean = tf.concat((mean,tf.reshape(meant,(1,batchSize,xDim))),0)
#    vart = tf.nn.softplus(Wvar*hCombined[-1]+bvar)
    vart = tf.nn.softplus(Wvar*hCombined+bvar)
#    var = tf.concat((var,vart),1)
    var = tf.concat((var,tf.reshape(vart,(1,batchSize,xDim))),0)
    zSampledt = tf.reduce_mean(tf.random_normal([10],meant,tf.sqrt(vart),dtype=tf.float32),
                               axis=1)
    zPrevious = tf.reshape(zSampledt,(batchSize,xDim))
#    zPrevious = meant
    
mean = mean[1:]
var = var[1:]

# ELBO calculation                                                                                      
S = 10              
#%% Term 1, the probability calculation only works for 1 demension
term1_1 = - nTimeSteps*tf.log(tf.sqrt(2*np.pi)*R)#.reshape((1)))
eps = tf.random_normal([S,nTimeSteps,batchSize,xDim],0,1)
#term1_2 = (xPlaceholder-B.reshape((1))*mean-B.reshape((1))*tf.sqrt(var)*eps)**2/(2*\
#          R.reshape((1))**2)
term1_2 = (xPlaceholder-B*mean-B*tf.sqrt(var)*eps)**2/(2*R**2)
term1_2 = tf.reduce_mean(term1_2,axis=0)
term1_2 = tf.reduce_sum(term1_2,axis=[0])

term1 = term1_1-term1_2

#%% Term 2
dist1 = tf.distributions.Normal(mean[0],var[0])
dist2 = tf.distributions.Normal(np.float32(0),np.float32(Q))
term2 = tf.distributions.kl_divergence(dist1,dist2,allow_nan_stats=False)

#%% Term 3
zt_1Sampled = mean[0:24]+eps[:,0:24]*var[0:24]
hCombinedts = 0.5*(tf.tanh(Wz*zt_1Sampled+bz)+hRight[1:])
meants = Wmu*hCombinedts + bmu
varts = tf.nn.softplus(Wvar*hCombinedts+bvar)
dist3 = tf.distributions.Normal(meants,varts)
dist4 = tf.distributions.Normal(A*zt_1Sampled,np.float32(Q))    
term3 = tf.distributions.kl_divergence(dist3,dist4)
term3 = tf.reduce_sum(tf.reduce_mean(term3,axis=0),axis = [0])
#%% loss
elbo = tf.reduce_mean(term1-term2-term3)


trainStep = tf.train.AdamOptimizer(0.001).minimize(loss=-elbo)
    
#%%

#def plot(loss_list, predictions_series, batchX, batchY):
#    plt.subplot(2, 3, 1)
#    plt.cla()
#    plt.plot(loss_list)
#
#    for batch_series_idx in range(5):
#        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
#        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
#
#        plt.subplot(2, 3, batch_series_idx + 2)
#        plt.cla()
#        plt.axis([0, truncated_backprop_length, 0, 2])
#        left_offset = range(truncated_backprop_length)
#        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
#        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
#        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")
#
#    plt.show()
#    plt.pause(0.0001)
#%%

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    allElbo = np.array([])

    for epoch_idx in range(num_epochs):
#        _current_state = np.zeros((batch_size, state_size))

        print("Epoch:", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchSize
            end_idx = start_idx + batchSize

            batchX = xTrain[start_idx:end_idx]
            batchX = np.swapaxes(batchX,0,1)
            batchX = np.flip(batchX,axis=0)
            
            batchZ = zTrain[start_idx:end_idx]
            
            _elbo,_trainStep = sess.run([elbo,trainStep],feed_dict={
                    xPlaceholder:batchX,
                    initState:np.zeros((batchSize,stateSize),dtype=np.float32),
                    z0:[0]})

            allElbo = np.append(allElbo,_elbo)

            if batch_idx%100 == 0:    
                print("Iteration:",epoch_idx*num_batches+batch_idx+1, "Elbo:", _elbo)
                fig = plt.figure(figsize=(15,5))
                ax1 = fig.add_subplot(1,2,1)
                plt.plot(allElbo)
                
                x_kalmanShape = np.reshape(xTest[0], [nTimeSteps, xDim])
                timeVec, testSmoothed = smoothKalman(x_kalmanShape, model)
                xTestBatch = xTest[0:batchSize]
                xTestBatch = np.swapaxes(xTestBatch,0,1)
                xTestBatch = np.flip(xTestBatch,axis=0)
                meanTest = sess.run(mean,feed_dict={
                        xPlaceholder: xTestBatch,
                        initState:np.zeros((batchSize,stateSize),dtype=np.float32),
                        z0:[0]})
                ax2 = fig.add_subplot(1,2,2)
                plt.scatter(np.arange(nTimeSteps),zTest[0],color='black')
                plt.plot(np.arange(nTimeSteps),meanTest[:,0],color='blue')
                plt.plot(np.arange(nTimeSteps),testSmoothed,color='red')
                plt.show()
#                plot(loss_list, _predictions_series, batchX, batchY)

#plt.ioff()
#plt.show()