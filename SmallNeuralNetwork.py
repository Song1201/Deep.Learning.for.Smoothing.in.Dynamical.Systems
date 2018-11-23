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

cnnPointEstimator = nn.CnnPointEstimator(200)



cnnPointEstimator.train(50,500,measure,hidden,'SmallNN/SmallNN.ckpt')


print("Training done!")
# saver.save(sess,'SmallNN/SmallNN.ckpt')
    
