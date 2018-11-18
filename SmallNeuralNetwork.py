# Seperate file for CNN point estimator. First make this file work, then 
# refactor it to small program structures.

#%%
import numpy as np
import tensorflow as tf
import GenerativeModel as gm
import matplotlib.pyplot as plt
import NeuralNetwork as nn

# #%%
# gm.linearGaussian(1,2,0,0.3,0,0.1,200,1000)

#%% Get data from generative model
hidden,measure = gm.loadData('Generated.Data/LG.1.2.0.0.3.0.0.1')
# Don't reshape hidden since it already has the same shape as output from the 
# network
measure = measure.reshape(-1,200,1)


#%%
# with tf.device('/GPU:0'):
#   measureP = tf.placeholder(dtype=tf.float32,shape=[None,200,1])
#   hiddenP = tf.placeholder(dtype=tf.float32,shape=[None,200])

#   with tf.variable_scope('Conv1',reuse=tf.AUTO_REUSE) as scope:
#     w = tf.get_variable(name='w',shape=[3,1,60],dtype=tf.float32,
#       initializer=tf.truncated_normal_initializer(stddev=0.1))
#     b = tf.get_variable(name='b',shape=[60],dtype=tf.float32,
#       initializer=tf.constant_initializer(0.1))
#     conv = tf.nn.convolution(measureP,w,dilation_rate=[1],padding='VALID')
#     preActivation = tf.nn.bias_add(conv,b)
#     conv1 = tf.nn.relu(preActivation)
#   with tf.variable_scope('FullyConnected',reuse=tf.AUTO_REUSE) as scope:
#     flatLength = conv1.shape[1].value * conv1.shape[2].value
#     flat = tf.reshape(conv1,[-1,flatLength])
#     w = tf.get_variable(name='w',shape=[flatLength,200],dtype=tf.float32,
#       initializer=tf.truncated_normal_initializer(stddev=0.1))
#     b = tf.get_variable(name='b',shape=[200],dtype=tf.float32,
#       initializer=tf.constant_initializer(0.1))
#     output = tf.matmul(flat,w) + b

#   loss = tf.reduce_sum(tf.sqrt(1+tf.square(hiddenP - output))-1)
#   train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# measureP,hiddenP,output,loss,train_step = nn.buildCnnPointEstimator(200)

cnnPointEstimator = nn.CnnPointEstimator(200)

saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(tf.initializers.global_variables())
  # sess.run(tf.initializers.local_variables())
  for i in range(1000):
    sess.run(cnnPointEstimator.trainStep,{cnnPointEstimator.measure:measure,
      cnnPointEstimator.hidden:hidden})
    if((i+1)%20==0): 
      print(sess.run(cnnPointEstimator.loss,{cnnPointEstimator.measure:measure,
        cnnPointEstimator.hidden:hidden}))

  path = saver.save(sess,'SmallNN/SmallNN.ckpt')
    
