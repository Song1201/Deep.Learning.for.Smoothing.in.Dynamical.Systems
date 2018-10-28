from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from smootherKalman import smoothKalman
from GenerativeModel import *

#%% Import and preprocess data
# x should be reversed and then send into the network!!!!!!
# x should be reversed and then send into the network!!!!!!
# x should be reversed and then send into the network!!!!!!


# Set the parameters for generative model
dim = 1
zDim = dim
xDim = dim
A = 0.9*np.identity(xDim)
B = 3.5*np.identity(xDim)
Q = np.identity(xDim)
R = Q
Q0 = Q

model = NLG(zDim, xDim, A, B, Q, R, 0, Q0)


#nTimeSteps = 25
xTest = np.load(r'C:\Project16\generateData\xTestNonLnGaus200.dat')
zTest = np.load(r'C:\Project16\generateData\zTestNonLnGaus200.dat')
xTest = xTest[:,0:25]
zTest = zTest[:,0:25]
nTimeSteps = xTest.shape[1]

    

#%% Set training parameters.
num_epochs = 100
stateSize = 8 # can be dicided arbitrarily
batchSize = 1

#%% Build the computational graph
xPlaceholder = tf.placeholder(tf.float32, [nTimeSteps,batchSize, xDim])

initState = tf.placeholder(tf.float32, [batchSize, stateSize])
# we want to output a number, so output size is 1
W = tf.Variable(np.random.rand(nTimeSteps,stateSize,1),dtype=tf.float32)
b = tf.Variable(np.zeros((nTimeSteps,1,1)), dtype=tf.float32)


cell = tf.nn.rnn_cell.BasicRNNCell(stateSize,activation=tf.nn.relu,reuse=None)

# Forward pass

# set time_major=True to speed up the computation, but need to transpose the input and output
# to be [nTimeSteps,batchSize,dim]

stateSeries,finalState = tf.nn.dynamic_rnn(cell,xPlaceholder,initial_state=initState,
                                                time_major=True) 
#%%   
hRight = tf.nn.relu(tf.matmul(stateSeries,W)+b)
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
S = 35              
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
zt_1Sampled = mean[0:-1]+eps[:,0:-1]*var[0:-1]
hCombinedts = 0.5*(tf.tanh(Wz*zt_1Sampled+bz)+hRight[1:])
meants = Wmu*hCombinedts + bmu
varts = tf.nn.softplus(Wvar*hCombinedts+bvar)
dist3 = tf.distributions.Normal(meants,varts)
dist4 = tf.distributions.Normal(A*zt_1Sampled,np.float32(Q))    
term3 = tf.distributions.kl_divergence(dist3,dist4)
term3 = tf.reduce_sum(tf.reduce_mean(term3,axis=0),axis = [0])
#%% loss
elbo = tf.reduce_mean(term1-term2-term3)


trainStep = tf.train.AdamOptimizer(0.0001).minimize(loss=-elbo)
    

#%%

iter_num = 30000
#zTest, xTest = model.generateSamples(nTimeSteps)
#zTest = np.reshape(zTest,(1,nTimeSteps,zDim))
#xTest = np.reshape(xTest,(1,nTimeSteps,xDim))
#xTest = np.load('xTest.dat')
#zTest = np.load('zTest.dat')
x_kalmanShape = np.reshape(xTest[0], [nTimeSteps, xDim])
timeVec, testSmoothed = smoothKalman(x_kalmanShape, model)
#%%
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#    init = tf.global_variables_initializer()
#    sess.run(init)
allElbo = np.array([])
for i in range(iter_num):
    batchX = xTest[0:1]
    batchX = np.swapaxes(batchX,0,1)
    batchX = np.flip(batchX,axis=0)            
        
    _elbo,_trainStep,_mean = sess.run([elbo,trainStep,mean],feed_dict={
            xPlaceholder:batchX,
            initState:np.zeros((batchSize,stateSize),dtype=np.float32),
            z0:[0]})

    allElbo = np.append(allElbo,_elbo)

    if i%100 == 0:    
        print("Iteration:",i+1, "Elbo:", _elbo)
        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(1,2,1)
        plt.plot(allElbo)
        
        ax2 = fig.add_subplot(1,2,2)
        plt.scatter(np.arange(nTimeSteps),zTest[0],color='black')
#        plt.plot(np.arange(nTimeSteps),np.flip(np.reshape(_mean,(-1)),0),color='blue')
        plt.plot(np.arange(nTimeSteps),np.reshape(_mean,(nTimeSteps)),color='blue')
        plt.plot(np.arange(nTimeSteps),testSmoothed,color='red')
        plt.show()
