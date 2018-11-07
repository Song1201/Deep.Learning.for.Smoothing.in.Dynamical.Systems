import numpy as np
import tensorflow as tf
#%%
xInput = tf.placeholder(tf.float32,(None,5))
term = xInput-1
output = []
for i in range(5):
    term = term
    
q