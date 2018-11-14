import numpy as np
import tensorflow as tf
from GenerativeModel import LG, NLG, LnonGaussian
from smootherKalman import smoothKalman
import matplotlib.pyplot as plt
#%%
'''
Methods
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv1d(x, W):#kasnke inte funkar. testa, annars läs dokumentationen
    return tf.nn.conv1d(x, W, 1, padding='VALID')

def convDillated1d(x, W, layer):
    conv = tf.nn.convolution(x, W, dilation_rate=[2**layer], padding='VALID')
    return conv

def returnNextBatch(batch_size, x, z):
    x_size = np.shape(x)[0]
    
    if x_size < batch_size:
        raise Exception("You are trying to get a batch that is larger than " +
                        "the amount of samples")
    
    idx = np.random.randint(x_size, size=batch_size)
    return x[idx], z[idx]
    
#def returnNextBatch(batch_size, nTimeSteps, model):
#    z = np.zeros([batch_size, nTimeSteps, dim])
#    x = np.zeros([batch_size, nTimeSteps, dim])
#    for i in range(batch_size):
#        z[i], x[i] = model.generateSamples(nTimeSteps)
#    
#    return z, x

    
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

model = LnonGaussian(zDim, xDim, A, B, Q, R, z0, Q0)

nTimeSteps = 200
nTrainSamples = 10000
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
zTrain = zTrain.reshape([nTrainSamples, nTimeSteps, dim])
xTrain = xTrain.reshape([nTrainSamples, nTimeSteps, dim])
zTest = zTest.reshape([nTestSamples, nTimeSteps, dim])
xTest = xTest.reshape([nTestSamples, nTimeSteps, dim])


'''
Neural net part
'''
#%%


observed = tf.placeholder(dtype = tf.float32, shape = [None, nTimeSteps, 1])
hidden = tf.placeholder(dtype=tf.float32, shape = [None, nTimeSteps, 1])
hidden_reshape = tf.reshape(hidden, [-1, nTimeSteps])

#shapen här är 3 - bredd på filtret, 1 - antal kanaler in, 60 - antal kanaler ut 
#(hur många olika filter vi applyar)
W_conv1 = weight_variable([3,1,60])#inte dilated nu men lägg in det sen
b_conv1 = bias_variable([60])

out1 = tf.nn.relu(conv1d(observed, W_conv1) + b_conv1)

#layer 2
W_conv2 = weight_variable([3,60,60])
b_conv2 = bias_variable([60])

out2 = tf.nn.relu(conv1d(out1, W_conv2) + b_conv2)

#layer 3
W_conv3 = weight_variable([3,60,60])
b_conv3 = bias_variable([60])

out3 = tf.nn.relu(convDillated1d(out2, W_conv3, 1))

#layer 4
W_conv4 = weight_variable([3,60,60])
b_conv4 = bias_variable([60])

out4 = tf.nn.relu(convDillated1d(out3, W_conv4, 2))

#layer 5
W_conv5 = weight_variable([3,60,60])
b_conv5 = bias_variable([60])

out5 = tf.nn.relu(convDillated1d(out4, W_conv5, 3))

#layer 6
W_conv6 = weight_variable([3,60,60])
b_conv6 = bias_variable([60])

out6 = tf.nn.relu(convDillated1d(out5, W_conv6, 4))

#layer 7
W_conv7 = weight_variable([3,60,60])
b_conv7 = bias_variable([60])

out7 = tf.nn.relu(convDillated1d(out6, W_conv7, 5))


#flat layer
'''
out_flat = tf.reshape(out2, [-1, nTimeSteps*60])

W_flat = weight_variable([nTimeSteps*60, nTimeSteps])
b_flat = bias_variable([nTimeSteps])

#output layer
output = tf.matmul(out_flat, W_flat) + b_flat
'''

out_flat = tf.reshape(out7, [-1, 72*60, 1])
W_convFinal = weight_variable([72*60, 1, nTimeSteps])
b_convFinal = bias_variable([nTimeSteps])

out_final = tf.nn.conv1d(out_flat, W_convFinal, 1, padding='VALID')

output = tf.reshape(out_final, [-1, nTimeSteps])

#loss function and optimization step
loss = tf.reduce_sum(tf.sqrt(1+tf.square(hidden_reshape - output)) -1)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss) #här måste vi nog speca lite mer

#call this to get for instance the loss from just input and output, to see if the whole smoother is any good.
loss_comp = tf.reduce_sum(tf.sqrt(1+tf.square(hidden - observed)) -1)

'''
sess = tf.InteractiveSession()#replace with tf.Session and add later
init = tf.global_variables_initializer()
sess.run(init)'''

#%% Training of the neural net
batch_size = 50
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(10000):
        batch_obs, batch_hidden = returnNextBatch(batch_size, xTrain, zTrain)
#        batch_obs, batch_hidden = returnNextBatch(batch_size, nTimeSteps, model)
        if i % 100 == 0:
            curr_loss = sess.run(loss, feed_dict={observed: batch_obs,
                                                  hidden: batch_hidden})
            print("step %d, loss: %g" % (i, curr_loss))
        train_step.run(feed_dict={observed: batch_obs,
                                                  hidden: batch_hidden})
    CNN_smoothed = sess.run(output, feed_dict={observed: xTest})

#%%
'''
Kalman smoother for reference - smoothes test data
'''
x_kalmanShape = np.reshape(xTest, [nTimeSteps, xDim])
timeVec, testSmoothed = smoothKalman(x_kalmanShape, model)
#%% plots
plt.plot(timeVec, testSmoothed, color = 'blue')
plt.scatter(timeVec, np.reshape(zTest, [nTimeSteps]), color = 'green', s = 5)
plt.plot(timeVec, np.reshape(CNN_smoothed, [nTimeSteps]), color = 'red')
plt.scatter(timeVec, np.reshape(xTest, [nTimeSteps]), color = 'black', s = 2)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
cnnTest = np.reshape(CNN_smoothed, [1, 200, 1])
kalTest = np.reshape(testSmoothed, [1, 200, 1])
sess.run(loss_comp, {hidden: zTest, observed: cnnTest})
sess.run(loss_comp, {hidden: zTest, observed: xTest})
sess.run(loss_comp, {hidden: zTest, observed: kalTest})
