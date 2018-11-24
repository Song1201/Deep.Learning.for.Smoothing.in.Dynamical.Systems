import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

class CnnPointEstimator:

  def __init__(self,numTimeSteps):
    self.measure,self.hidden,self.output,self.loss,self.trainStep = \
      self._buildCnnPointEstimator(numTimeSteps)

  def train(self,batchSize,numEpochs,measure,hidden,savePath,testMeasure,
    testHidden,testSample):
    # measure and hidden are 2-D numpy arrays, whose shape[1] is the number of 
    # time steps.
    saver = tf.train.Saver()
    measure = measure.reshape(-1,200,1)
    testMeasure = testMeasure.reshape(-1,200,1)
    # A index trick to make the one layer slice has the same number of dimension
    # as before. 
    sampleTestMeasure = testMeasure[testSample:testSample+1,:,:]
    sampleTestHidden = testHidden[testSample:testSample+1]
    np.random.shuffle(measure)
    numIters = measure.shape[0]//batchSize
    loss = np.zeros([numIters*numEpochs])

    with tf.Session() as sess:
      sess.run(tf.initializers.global_variables())
      iterRun = 0 # How many iteration has been run during training
      for i in range(numEpochs):
        for j in range(numIters):
          batchMeasure = measure[j*batchSize:(j+1)*batchSize]
          batchHidden = hidden[j*batchSize:(j+1)*batchSize]
          loss[iterRun],dump = sess.run([self.loss,self.trainStep],
            {self.measure:batchMeasure,self.hidden:batchHidden})
          iterRun += 1

        if (i+1)%10==0:    
          print('Epoch: '+str(i+1)+'  Train loss: '+str(loss[iterRun-1]))
          plt.plot(loss[:iterRun-1],color='blue')
          plt.show()

          print('Epoch: '+str(i+1)+'  Test loss: '+str(self.loss.eval(
            {self.measure:testMeasure,self.hidden:testHidden})))
          sampleTestOutput = self.output.eval({self.measure:sampleTestMeasure,
            self.hidden:sampleTestHidden})
          plt.scatter(np.arange(self.measure.shape[1].value),sampleTestMeasure,
            marker='o',color='green',s=0.3)
          plt.plot(sampleTestHidden.flatten(),color='blue')
          plt.plot(sampleTestOutput.flatten(),color='red')
          plt.show()
      
      saver.save(sess,savePath)
    
    loss.dump(savePath[:-4]+'loss')

  def infer(self,measure,hidden,variablePath):
    # measure and hidden are 2-D numpy arrays, whose shape[1] is the number of 
    # time steps.   
    measure = measure.reshape(-1,200,1) 
    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess,variablePath)

      return sess.run(self.output,{self.measure:measure}),\
        sess.run(self.loss,{self.measure:measure,self.hidden:hidden})

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
      measure = tf.placeholder(dtype=DTYPE,
        shape=[None,numTimeSteps,NUM_DATA_DIM])
      hidden = tf.placeholder(dtype=DTYPE,shape=[None,numTimeSteps])

      with tf.variable_scope('Conv1',reuse=tf.AUTO_REUSE) as scope:
        w = tf.get_variable(name='w',shape=[KERNEL_SIZE,NUM_DATA_DIM,NUM_KERNEL],
          dtype=DTYPE,
          initializer=tf.truncated_normal_initializer(stddev=INIT_STD))
        b = tf.get_variable(name='b',shape=[NUM_KERNEL],dtype=DTYPE,
          initializer=tf.constant_initializer(INIT_CONST))
        conv = tf.nn.convolution(measure,w,dilation_rate=[1],padding='VALID')
        preActivation = tf.nn.bias_add(conv,b)
        conv1 = tf.nn.relu(preActivation)
      with tf.variable_scope('FullyConnected',reuse=tf.AUTO_REUSE) as scope:
        flatLength = conv1.shape[1].value * conv1.shape[2].value
        flat = tf.reshape(conv1,[-1,flatLength])
        w = tf.get_variable(name='w',shape=[flatLength,numTimeSteps],
          dtype=DTYPE,
          initializer=tf.truncated_normal_initializer(stddev=INIT_STD))
        b = tf.get_variable(name='b',shape=[numTimeSteps],dtype=DTYPE,
          initializer=tf.constant_initializer(INIT_CONST))
        output = tf.matmul(flat,w) + b

      # loss defined according to report
      loss = tf.reduce_sum(tf.sqrt(1+tf.square(hidden - output))-1)
      trainStep = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return measure, hidden, output, loss, trainStep