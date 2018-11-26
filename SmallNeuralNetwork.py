# Seperate file for CNN point estimator. First make this file work, then 
# refactor it to small program structures.

#%%
import numpy as np
import tensorflow as tf
import GenerativeModel as gm
import matplotlib.pyplot as plt
import NeuralNetwork as nn

#%% Get data from generative model
hidden,measure = gm.loadData('Generated.Data/LG.Train.0.9.3.5.0.1.0.1')
testHidden, testMeasure = gm.loadData('Generated.Data/LG.Test.0.9.3.5.0.1.0.1')
cnnPointEstimator = nn.CnnPointEstimator(200)


#%%
cnnPointEstimator.train(2e-4,100,200,measure,hidden,'SmallNN/SmallNN.ckpt',
  testMeasure,testHidden,20)


print("Training done!")   