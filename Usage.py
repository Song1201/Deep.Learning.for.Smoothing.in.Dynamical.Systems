# This file shows how to use the models to reproduce the project. Please run 
# this file using Jupyter Notebook or other tools that can run each cell 
# seperately to see its function. See cell's comments for detailed description.

#%% Generate data
import GenerativeModel as gm
# Training data
gm.linearGaussian(0.9,3.5,0,1,0,1,200,20000)
# Testing data
gm.linearGaussian(0.9,3.5,0,1,0,1,200,5000)

#%% Process the data with Kalman smoother.
import KalmanSmoother as ks
import GenerativeModel as gm
import numpy as np
import matplotlib.pyplot as plt

testHidden, testMeasure = gm.loadData('Generated.Data/LG.Test.0.9.3.5.0.1.0.1')
numTimeSteps = testHidden.shape[1]
smoother = ks.KalmanSmoother(0.9,3.5,0,1,0,1)
sampleNo = 456

smoother.smooth(testMeasure,'Kalman.Results')
testKalmanMean,testKalmanStd = ks.loadResults('Kalman.Results')

plt.figure(figsize=(10,5))
plt.scatter(np.arange(numTimeSteps),testHidden[sampleNo],marker='o',
  color='blue',s=4)
plt.plot(testKalmanMean[sampleNo].flatten(),color='green')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(testKalmanStd,color='green')
plt.show()
