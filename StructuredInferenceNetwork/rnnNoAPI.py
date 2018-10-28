import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%% Import and preprocess data
xTest = np.load(r'C:\Project16\generateData\LG200xTest.dat')
zTest = np.load(r'C:\Project16\generateData\LG200zTest.dat')
xTrain = np.load(r'C:\Project16\generateData\LG200xTrain.dat')
zTrain = np.load(r'C:\Project16\generateData\LG200zTrain.dat')
# defaulted x,z has the same number of time steps
nTimeSteps = xTest.shape[1]
xDim = xTest.shape[2]
# Set the parameters for generative model
A = 0.9*np.identity(xDim)
B = 3.5*np.identity(xDim)
Q = np.identity(xDim)
R = Q
Q0 = Q

#%% This network is only disigned for 1-d data.
xInput = tf.placeholder(tf.float32, [None,nTimeSteps])
#batchSize = xPlaceholder.shape[0]
hiddenSize = 5
wHidden = []
bHidden = []
wOutput = []
bOutput = []
for t in range(nTimeSteps):
    wHidden.append(tf.Variable(np.random.normal(0,0.1,(2,hiddenSize)),dtype=tf.float32))
    bHidden.append(tf.Variable(np.random.normal(0,0.1,(hiddenSize)),dtype=tf.float32))
    wOutput.append(tf.Variable(np.random.normal(0,0.1,(hiddenSize,1)),dtype=tf.float32))
    bOutput.append(tf.Variable(np.random.normal(0,0.1,(1)),dtype=tf.float32))

initState = tf.placeholder(tf.float32,(None)) 
#%%   
inputAndState = tf.stack((xInput[:,-1],initState),axis=1)
# where I stop 
for t in range(1,nTimeSteps+1):
    nextState1 = tf.nn.leaky_relu(tf.matmul(inputAndState,wHidden[-t])+bHidden[-t])
    nextState = tf.nn.leaky_relu(tf.matmul(nextState1,wOutput[-t]))

sess.run(inputAndState,feed_dict={xInput:np.ones((3,200)),initState:np.zeros((3))})
hRight = tf.nn.relu(tf.reshape(tf.matmul(nextState,W[-1]),(xDim,batchSize))+b[-1])
hRight = tf.reshape(hRight,(1,xDim,batchSize)) 

for t in range(2,nTimeSteps+1):
    inputAndState = tf.concat((tf.reshape(tf.transpose(obSeries[-t],(1,0)),(xDim,batchSize,1)),
                          nextState),axis=2)
    nextState = tf.nn.relu(tf.matmul(inputAndState,wRNN[-t])+bRNN[-t])
    
    hRightNew = tf.nn.relu(tf.reshape(tf.matmul(nextState,W[-t]),(xDim,batchSize))+b[-t]) 
    hRight = tf.concat((hRight,tf.reshape(hRightNew,(1,xDim,batchSize))),axis=0)
    

#%%   
hRight = tf.transpose(hRight,perm=(0,2,1))
hRight = tf.reverse(hRight,axis=[0])                                        

z0 = tf.placeholder(tf.float32,[zDim])
Wz = tf.Variable(np.random.rand(1),dtype=tf.float32)
bz = tf.Variable(0.1*np.zeros((1)),dtype=tf.float32)
Wmu = tf.Variable(np.random.rand(1),dtype=tf.float32)
bmu = tf.Variable(0.1*np.zeros((1)),dtype=tf.float32)
Wvar = tf.Variable(np.random.rand(1),dtype=tf.float32)
bvar = tf.Variable(0.1*np.zeros((1)),dtype=tf.float32)
mean = tf.zeros((1,batchSize,xDim))
var = tf.zeros((1,batchSize,xDim)) # variance
#zSampled = tf.zeros((batchSize,xDim))
zPrevious = tf.reshape(tf.tile(z0,[batchSize]),(batchSize,xDim))

for t in range(nTimeSteps):
    hCombined = 0.5*(tf.tanh(Wz*zPrevious+bz)+hRight[t])
    meant = Wmu*hCombined + bmu
#    mean = tf.concat((mean,meant),1)
    mean = tf.concat((mean,tf.reshape(meant,(1,batchSize,xDim))),0)
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


train_step = tf.train.AdamOptimizer(0.001).minimize(loss=-elbo)
    
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
            
            _elbo, _train_step,_mean,_hRight = sess.run(
                [elbo, train_step,mean,hRight],
                feed_dict={
                    xPlaceholder:batchX,
                    z0:[0]
                })

            allElbo = np.append(allElbo,_elbo)

            if batch_idx%100 == 0:    
                print("Step:",batch_idx, "Elbo:", _elbo)
                fig = plt.figure(figsize=(15,5))
                ax1 = fig.add_subplot(1,2,1)
                plt.plot(allElbo)
                ax2 = fig.add_subplot(1,2,2)
                plt.scatter(np.arange(nTimeSteps),batchZ[0])
                plt.plot(np.arange(nTimeSteps),_mean[:,0])
                plt.show()
#                plot(loss_list, _predictions_series, batchX, batchY)

#plt.ioff()
#plt.show()