import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

class CnnPointEstimator:

  def __init__(self,numTimeSteps):
    # self.measure,self.hidden,self.output = self._buildCnnPointEstimator(
    #   numTimeSteps)
    self._buildCnnPointEstimator(numTimeSteps)

  def train(self,lr,batchSize,numEpochs,measure,hidden,savePath,testMeasure,
    testHidden,testSample):
    with tf.device('/GPU:0'):
      trainStep = tf.train.AdamOptimizer(learning_rate=lr).minimize(
        self.loss)
    
    # measure and hidden are 2-D numpy arrays, whose shape[1] is the number of 
    # time steps.
    saver = tf.train.Saver()
    measure = measure.reshape(-1,200,1)
    testMeasure = testMeasure.reshape(-1,200,1)
    # A index trick to make the one layer slice has the same number of dimension
    # as before. 
    sampleTestMeasure = testMeasure[testSample:testSample+1,:,:]
    sampleTestHidden = testHidden[testSample:testSample+1]
    numIters = measure.shape[0]//batchSize
    allLoss = np.zeros([numIters*numEpochs])
    iterRun = 0 # How many iteration has been run during training
    randIndex = np.arange(measure.shape[0])

    with tf.Session() as sess:
      sess.run(tf.initializers.global_variables())
      for i in range(numEpochs):
        np.random.shuffle(randIndex)
        measure = measure[randIndex] 
        hidden = hidden[randIndex]        
        for j in range(numIters):
          batchMeasure = measure[j*batchSize:(j+1)*batchSize]
          batchHidden = hidden[j*batchSize:(j+1)*batchSize]
          allLoss[iterRun],dump = sess.run([self.loss,trainStep],
            {self.measure:batchMeasure,self.hidden:batchHidden})
          iterRun += 1

        if (i+1)%1==0:    
          print(dt.now().time())
          # Mean of all losses of batches in this epoch
          meanLossEpoch = np.mean(allLoss[i*numIters:(i+1)*numIters])
          print('Epoch: '+str(i+1)+'  Train loss: '+str(meanLossEpoch))
          plt.figure(figsize=(10,5))
          plt.plot(allLoss[:iterRun-1],color='blue')
          plt.show()

          print('Epoch: '+str(i+1)+'  Test loss: '+str(self.loss.eval(
            {self.measure:testMeasure,self.hidden:testHidden})))
          sampleTestOutput = self.output.eval({self.measure:sampleTestMeasure,
            self.hidden:sampleTestHidden})
          plt.figure(figsize=(10,5))
          plt.scatter(np.arange(self.measure.shape[1].value),sampleTestMeasure,
            marker='o',color='green',s=0.3)
          plt.plot(sampleTestHidden.flatten(),color='blue')
          plt.plot(sampleTestOutput.flatten(),color='red')
          plt.show()
      
      saver.save(sess,savePath)
    
    allLoss.dump(savePath[:-4]+'loss')

  def infer(self,measure,hidden,variablePath):
    # measure and hidden are 2-D numpy arrays, whose shape[1] is the number of 
    # time steps.   
    measure = measure.reshape(-1,200,1) 
    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess,variablePath)

      return sess.run([self.output,self.loss],{self.measure:measure})

  # Build a graph for CNN point estimator. Return the placeholder for input, and 
  # other necessary interface.
  def _buildCnnPointEstimator(self,numTimeSteps):
    KERNEL_SIZE = 3
    NUM_KERNEL = 60
    INIT_STD = 0.1
    INIT_CONST = 0.1
    NUM_DATA_DIM = 1 # number of data dimensions
    DTYPE = tf.float32

    with tf.device('/GPU:0'):
      self.measure = tf.placeholder(dtype=DTYPE,
        shape=[None,numTimeSteps,NUM_DATA_DIM])
      self.hidden = tf.placeholder(dtype=DTYPE,shape=[None,numTimeSteps])

      # with tf.variable_scope('Conv1',reuse=tf.AUTO_REUSE) as scope:
      #   w = tf.get_variable(name='w',shape=[KERNEL_SIZE,NUM_DATA_DIM,NUM_KERNEL],
      #     dtype=DTYPE,
      #     initializer=tf.truncated_normal_initializer(stddev=INIT_STD))
      #   b = tf.get_variable(name='b',shape=[NUM_KERNEL],dtype=DTYPE,
      #     initializer=tf.constant_initializer(INIT_CONST))
      #   conv = tf.nn.convolution(measure,w,dilation_rate=[1],padding='VALID')
      #   preActivation = tf.nn.bias_add(conv,b)
      #   conv1 = tf.nn.relu(preActivation)

      # with tf.variable_scope('Conv2',reuse=tf.AUTO_REUSE) as scope:
      #   w = tf.get_variable(name='w',shape=[KERNEL_SIZE,60,NUM_KERNEL],
      #     dtype=DTYPE,
      #     initializer=tf.truncated_normal_initializer(stddev=INIT_STD))
      #   b = tf.get_variable(name='b',shape=[NUM_KERNEL],dtype=DTYPE,
      #     initializer=tf.constant_initializer(INIT_CONST))
      #   conv = tf.nn.convolution(conv1,w,dilation_rate=[1],padding='VALID')
      #   preActivation = tf.nn.bias_add(conv,b)
      #   conv2 = tf.nn.relu(preActivation)

      with tf.variable_scope('Conv1',reuse=tf.AUTO_REUSE) as scope:
        conv1 = self._dilatedConv(KERNEL_SIZE,self.measure,NUM_KERNEL,DTYPE,
          INIT_STD,INIT_CONST,1)

      with tf.variable_scope('Conv2',reuse=tf.AUTO_REUSE) as scope:    
        conv2 = self._dilatedConv(KERNEL_SIZE,conv1,NUM_KERNEL,DTYPE,INIT_STD,
          INIT_CONST,1)

      with tf.variable_scope('Conv3',reuse=tf.AUTO_REUSE) as scope:    
        conv3 = self._dilatedConv(KERNEL_SIZE,conv2,NUM_KERNEL,DTYPE,INIT_STD,
          INIT_CONST,2)

      with tf.variable_scope('Conv4',reuse=tf.AUTO_REUSE) as scope:    
        conv4 = self._dilatedConv(KERNEL_SIZE,conv3,NUM_KERNEL,DTYPE,INIT_STD,
          INIT_CONST,4)

      with tf.variable_scope('Conv5',reuse=tf.AUTO_REUSE) as scope:    
        conv5 = self._dilatedConv(KERNEL_SIZE,conv4,NUM_KERNEL,DTYPE,INIT_STD,
          INIT_CONST,8)
      
      with tf.variable_scope('Conv6',reuse=tf.AUTO_REUSE) as scope:    
        conv6 = self._dilatedConv(KERNEL_SIZE,conv5,NUM_KERNEL,DTYPE,INIT_STD,
          INIT_CONST,16)
      
      with tf.variable_scope('FinalConv',reuse=tf.AUTO_REUSE) as scope:    
        finalConv = self._dilatedConv(KERNEL_SIZE,conv6,NUM_KERNEL,DTYPE,
          INIT_STD,INIT_CONST,32)

      with tf.variable_scope('FullyConnected',reuse=tf.AUTO_REUSE) as scope:
        flatLength = finalConv.shape[1].value * finalConv.shape[2].value
        flat = tf.reshape(finalConv,[-1,flatLength])
        w = tf.get_variable(name='w',shape=[flatLength,numTimeSteps],
          dtype=DTYPE,
          initializer=tf.truncated_normal_initializer(stddev=INIT_STD))
        b = tf.get_variable(name='b',shape=[numTimeSteps],dtype=DTYPE,
          initializer=tf.constant_initializer(INIT_CONST))
        self.output = tf.matmul(flat,w) + b

      # # loss defined according to report
      self.loss = tf.reduce_sum(tf.sqrt(1+tf.square(self.hidden-self.output))-1)
      # trainStep = tf.train.AdamOptimizer(2e-4).minimize(loss)

    # return measure, hidden, output, loss, trainStep
    # return measure, hidden, output, 

  def _dilatedConv(self,kernelSize,inputTensor,numKernel,dtype,
    initStd,initConst,dilatedRate):
    w = tf.get_variable(name='w',shape=[kernelSize,inputTensor.shape[2].value,
      numKernel],dtype=dtype,
      initializer=tf.truncated_normal_initializer(stddev=initStd))
    b = tf.get_variable(name='b',shape=[numKernel],dtype=dtype,
      initializer=tf.constant_initializer(initConst))
    conv = tf.nn.convolution(inputTensor,w,dilation_rate=[dilatedRate],
      padding='VALID')
    return tf.nn.relu(tf.nn.bias_add(conv,b))    