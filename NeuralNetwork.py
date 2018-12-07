import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import KalmanSmoother as ks

class CnnPointEstimator:

  def __init__(self,numTimeSteps):
    self._buildCnnPointEstimator(numTimeSteps)

  def train(self,lr,batchSize,numEpochs,measure,hidden,savePath,testMeasure,
    testHidden,testSample,showKalman=False):
    # measure and hidden are 2-D numpy arrays, whose shape[1] is the number of 
    # time steps.    
    with tf.device('/GPU:0'):
      trainStep = tf.train.AdamOptimizer(learning_rate=lr).minimize(
        self.loss)
    # A index trick to make the one layer slice has the same number of dimension
    # as before. 
    sampleTestMeasure = testMeasure[testSample:testSample+1]
    sampleTestHidden = testHidden[testSample:testSample+1]
    if showKalman:
      testKalmanZ,dump = ks.loadResults('Results.Data/LG.Kalman.Results')
      sampleTestKalmanZ = testKalmanZ[testSample]
    numIters = measure.shape[0]//batchSize
    allLoss = np.zeros([numIters*numEpochs])
    iterRun = 0 # How many iteration has been run during training
    randIndex = np.arange(measure.shape[0])
    
    saver = tf.train.Saver()
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
          print('Epoch: '+str(i+1)+'  Train loss per batch: ' + 
            str(meanLossEpoch))
          plt.figure(figsize=(10,5))
          plt.plot(allLoss[:iterRun-1],color='blue')
          plt.show()

          print('Epoch: '+str(i+1)+'  Test loss per batch: '+str(self.loss.eval(
            {self.measure:testMeasure,self.hidden:testHidden})/
            testHidden.shape[0]*batchSize))
          sampleTestOutput = self.output.eval({self.measure:sampleTestMeasure,
            self.hidden:sampleTestHidden})
          plt.figure(figsize=(10,5))
          plt.scatter(np.arange(sampleTestHidden.shape[1]),sampleTestHidden,
            marker='o',color='blue',s=4)
          if showKalman: plt.plot(sampleTestKalmanZ,color='green')
          plt.plot(sampleTestOutput.flatten(),color='red')
          plt.show()
      
      saver.save(sess,savePath)
    
    allLoss.dump(savePath[:-4]+'loss')
    print('Training Done!')

  def infer(self,measure,variablePath): 
    if len(measure.shape)==1: 
      measure = measure.reshape(1,-1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess,variablePath)
      return self.output.eval({self.measure:measure})

  def computeLoss(self,hypothesis,trueValue,variablePath):
    if len(hypothesis.shape)==1: hypothesis = hypothesis.reshape(1,-1)
    if len(trueValue.shape)==1: trueValue = trueValue.reshape(1,-1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess,variablePath)
      return self.loss.eval({self.output:hypothesis,self.hidden:trueValue})

  # Build a graph for CNN point estimator. Return the placeholder for input, and 
  # other necessary interface.
  def _buildCnnPointEstimator(self,numTimeSteps):
    KERNEL_SIZE = 3
    NUM_KERNEL = 60
    INIT_STD = 0.1
    INIT_CONST = 0.1
    DTYPE = tf.float32

    with tf.device('/GPU:0'):
      self.measure = tf.placeholder(dtype=DTYPE,shape=[None,numTimeSteps])
      measure = tf.reshape(self.measure,[-1,numTimeSteps,1])
      self.hidden = tf.placeholder(dtype=DTYPE,shape=[None,numTimeSteps])

      with tf.variable_scope('Conv1',reuse=tf.AUTO_REUSE) as scope:
        conv1 = self._dilatedConv(KERNEL_SIZE,measure,NUM_KERNEL,DTYPE,
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