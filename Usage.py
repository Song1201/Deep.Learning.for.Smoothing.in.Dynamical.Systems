# This file shows how to use the models to reproduce the project. Please run 
# this file using Jupyter Notebook or other tools that can run each cell 
# seperately to see its function. See cell's comments for detailed description.

#%% Generate data
import GenerativeModel as gm
# Training data
gm.linearGaussian(0.9,3.5,0,1,0,1,200,20000)
# Testing data
gm.linearGaussian(0.9,3.5,0,1,0,1,200,5000)

#%% Process the data with Kalman smoother
import GenerativeModel as gm
import KalmanSmoother as ks
import numpy as np

hidden,measure = gm.loadData('Generated.Data/LG.Train.0.9.3.5.0.1.0.1')
testHidden, testMeasure = gm.loadData('Generated.Data/LG.Test.0.9.3.5.0.1.0.1')
numTimeSteps = hidden.shape[1]
kf = ks.KalmanSmoother(0.9,3.5,0,1,0,1)
testKalmanMean, testKalmanStd = kf.smooth(testMeasure)
print(testKalmanMean.shape)
print(testKalmanStd.shape)
plt.show()