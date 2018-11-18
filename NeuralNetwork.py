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
    