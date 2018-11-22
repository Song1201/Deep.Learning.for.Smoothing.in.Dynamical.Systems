import tensorflow as tf

# Build a graph for CNN point estimator. Return the placeholder for input, and 
# other necessary interface.
def _buildCnnPointEstimator(numTimeSteps):
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

class CnnPointEstimator:
  def __init__(self,numTimeSteps):
    self.measure,self.hidden,self.output,self.loss,self.trainStep = \
      _buildCnnPointEstimator(numTimeSteps)

  def train(self,numSteps,measure,hidden,savePath):
    # measure and hidden are 2-D numpy arrays, whose shape[1] is the number of 
    # time steps.
    saver = tf.train.Saver()

    with tf.device('/GPU:0'):
      measure = tf.constant(measure.reshape(-1,200,1))

    with tf.Session() as sess:
      sess.run(tf.initializers.global_variables())
      for i in range(numSteps):
        sess.run(self.trainStep,{self.measure:measure,self.hidden:hidden})
        if((i+1)%50==0): 
          print(sess.run(self.loss,{self.measure:measure,self.hidden:hidden})) 
      saver.save(sess,savePath)

  def infer(self,measure,hidden,variablePath):
    # measure and hidden are 2-D numpy arrays, whose shape[1] is the number of 
    # time steps.   
    measure = measure.reshape(-1,200,1) 
    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess,variablePath)

      return sess.run(self.output,{self.measure:measure}),\
        sess.run(self.loss,{self.measure:measure,self.hidden:hidden})
