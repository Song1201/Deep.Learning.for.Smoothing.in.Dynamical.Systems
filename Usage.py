# This file shows how to use the models to reproduce the project. Please run 
# this file using Jupyter Notebook or other tools that can run each cell 
# separately to see its function. See cell's comments for detailed description.

#%% Generate data from linear Gaussian model
import GenerativeModel as gm
gm.linearGaussian(200,5000)

#%% Generate data from non-linear Gaussian model
import GenerativeModel as gm
gm.nonLinearGaussian(200,5000)

#%% Generate data from linear non-Gaussian model
import GenerativeModel as gm
gm.linearNonGaussian(200,5000)

#%% Generate data from non-linear non-Gaussian model
import GenerativeModel as gm
gm.nonLinearNonGaussian(200,5000)

#%% Process data from linear Gaussian model with Kalman smoother.
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

#%% Build and train a CNN point estimator for linear Gaussian model
import GenerativeModel as gm
import NeuralNetwork as nn

hidden,measure = gm.loadData('Generated.Data/LG.Train.0.9.3.5.0.1.0.1')
testHidden, testMeasure = gm.loadData('Generated.Data/LG.Test.0.9.3.5.0.1.0.1')
cnnPointEstimator = nn.CnnPointEstimator(hidden.shape[1])
cnnPointEstimator.train(2e-4,100,200,measure,hidden,
  'Trained.Models/CNN.Point.Estimator.LG.ckpt',testMeasure,testHidden,1213)

#%% Load a CNN point estimator and test it.
import numpy as np
import GenerativeModel as gm;
import matplotlib.pyplot as plt
import NeuralNetwork as nn
import KalmanSmoother as ks

testHidden,testMeasure = gm.loadData('Generated.Data/LG.Test.0.9.3.5.0.1.0.1')
cnnPointEstimator = nn.CnnPointEstimator(testHidden.shape[1])
sampleNo = 1213
estimated = cnnPointEstimator.infer(testMeasure[sampleNo],
  'Trained.Models/CNN.Point.Estimator.LG/CNN.Point.Estimator.LG.ckpt')
loss = cnnPointEstimator.computeLoss(estimated,testHidden[sampleNo],
  'Trained.Models/CNN.Point.Estimator.LG/CNN.Point.Estimator.LG.ckpt')
testKalmanZ,dump = ks.loadResults('Results.Data/LG.Kalman.Results')
plt.figure(figsize=(10,5))
plt.scatter(np.arange(testHidden.shape[1]),testHidden[sampleNo],
  marker='o',color='blue',s=4)
plt.plot(testKalmanZ[sampleNo],color='green')
plt.plot(estimated.flatten(),color='red')
plt.show()
