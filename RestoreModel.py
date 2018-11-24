#%%
import numpy as np
import tensorflow as tf
import GenerativeModel as gm;
import matplotlib.pyplot as plt
import NeuralNetwork as nn

#%%
hidden,measure = gm.loadData('Generated.Data/LG.Train.0.9.3.5.0.1.0.1')


# measureP,hiddenP,output,loss,train_step = nn.buildCnnPointEstimator(200)
cnnPointEstimator = nn.CnnPointEstimator(200)

#%%
output,loss = cnnPointEstimator.infer(measure,hidden,'SmallNN/SmallNN.ckpt')
print(output.shape)
print(loss)