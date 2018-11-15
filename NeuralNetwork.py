import numpy as np
import tensorflow as tf
from smootherKalman import smoothKalman
import matplotlib.pyplot as plt
#%%
def dilatedConvLayer(kernelLength,numKernel,dilatFactor,x):
	weightShape = kernelShape.copy()
	weightShape.append(numKernel)
	w = tf.Variable(tf.truncated_normal([kernelLength,x.shape[2].value,numKernel],
    stddev=0.1))
	b = tf.Variable(tf.constant(0.1,shape=[numKernel]))
	return tf.nn.convolution(x,w,dilation_rate=[dilatFactor],padding='VALID') + b

def buildCnnPointEstimator(numTimeSteps) {
  x = tf.placeholder(dtype = tf.float32, shape = [None, nTimeSteps, 1])

  conv1 = tf.layers.conv1d(inputs=x,filters=60,kernel_size=3,dilation_rate=1,
    activation=tf.nn.relu,use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),
    bias_initializer=tf.constant_initializer(value=0.1))
  conv2 = tf.layers.conv1d(inputs=conv1,filters=60,kernel_size=3,
    dilation_rate=1,activation=tf.nn.relu,use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),
    bias_initializer=tf.constant_initializer(value=0.1))
  conv3 = tf.layers.conv1d(inputs=conv2,filters=60,kernel_size=3,
    dilation_rate=2,activation=tf.nn.relu,use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),
    bias_initializer=tf.constant_initializer(value=0.1))
  conv4 = tf.layers.conv1d(inputs=conv3,filters=60,kernel_size=3,
    dilation_rate=4,activation=tf.nn.relu,use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),
    bias_initializer=tf.constant_initializer(value=0.1))
  conv5 = tf.layers.conv1d(inputs=conv4,filters=60,kernel_size=3,
    dilation_rate=8,activation=tf.nn.relu,use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),
    bias_initializer=tf.constant_initializer(value=0.1))
  conv6 = tf.layers.conv1d(inputs=conv5,filters=60,kernel_size=3,
    dilation_rate=16,activation=tf.nn.relu,use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),
    bias_initializer=tf.constant_initializer(value=0.1))
  conv7 = tf.layers.conv1d(inputs=conv6,filters=60,kernel_size=3,
    dilation_rate=32,activation=tf.nn.relu,use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1)
    bias_initializer=tf.constant_initializer(value=0.1))
  conv7Flat = tf.layers.flatten(conv7)
  output = tf.layers.dense(inputs=conv7Flat,units=numTimeSteps,use_bias=True,
    kernel_initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1),
    bias_initializer=tf.constant_initializer(value=0.1))
  
  return x, output
  # with tf.variable_scope("Conv1"):
  #   out1 = tf.nn.relu(dilatedConvLayer(KERNEL_LENGTH,numKernel=60,dilatFactor=1,
  #     x))
  # with tf.variable_scope("Conv2"):
  #   out2 = tf.nn.relu(dilatedConvLayer(KERNEL_LENGTH,numKernel=60,dilatFactor=1,
  #   out1))
  # with tf.variable_scope("Conv3"):
  #   out3 = tf.nn.relu(dilatedConvLayer(KERNEL_LENGTH,numKernel=60,dilatFactor=2,
  #   out2))
  # with tf.variable_scope("Conv4"):
  #   out4 = tf.nn.relu(dilatedConvLayer(KERNEL_LENGTH,numKernel=60,dilatFactor=4,
  #   out3))
  # with tf.variable_scope("Conv5"):
  #   out5 = tf.nn.relu(dilatedConvLayer(KERNEL_LENGTH,numKernel=60,dilatFactor=8,
  #   out4))
  # with tf.variable_scope("Conv6"):
  #   out6 = tf.nn.relu(dilatedConvLayer(KERNEL_LENGTH,numKernel=60,
  #     dilatFactor=16,out5))
  # with tf.variable_scope("Conv7"):
  #   out7 = tf.nn.relu(dilatedConvLayer(KERNEL_LENGTH,numKernel=60,
  #     dilatFactor=32,out6))
  # with tf.variable_scope("FullyConnect"):
  #   flatLength = out7.shape[1].value * out7.shape[2].value
  #   outFlat = tf.reshape(out7,[-1,flatLength,1])
}

def trainCnnPointEstimator() {
  
}

def returnNextBatch(batch_size, x, z):
    x_size = np.shape(x)[0]
    
    if x_size < batch_size:
        raise Exception("You are trying to get a batch that is larger than " +
                        "the amount of samples")
    
    idx = np.random.randint(x_size, size=batch_size)
    return x[idx], z[idx]

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
