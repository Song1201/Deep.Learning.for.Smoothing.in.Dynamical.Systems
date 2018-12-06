# Some testing code for this project.

#%% Test Generative Model linear Gaussian function
import GenerativeModel as gm
import Visualization as vs

gm.linearGaussian(200,20)
z,x = gm.loadData('Generated.Data/LG')
vs.scatterTimeSeries(z,'blue')
vs.scatterTimeSeries(x,'green')

#%% Test Generative Model non-linear Gaussian function
import GenerativeModel as gm
import Visualization as vs

gm.nonLinearGaussian(200,20)
z,x = gm.loadData('Generated.Data/NLG')
vs.scatterTimeSeries(z,'blue')
vs.scatterTimeSeries(x,'green')

#%% Test Generative Model linear non-Gaussian function
import GenerativeModel as gm
import Visualization as vs

gm.linearNonGaussian(200,20)
z,x = gm.loadData('Generated.Data/LNG')
vs.scatterTimeSeries(z,'blue')
vs.scatterTimeSeries(x,'green')

#%% Test Generative Model non-linear non-Gaussian function
import GenerativeModel as gm
import Visualization as vs

gm.nonLinearNonGaussian(200,20)
z,x = gm.loadData('Generated.Data/NLNG')
vs.scatterTimeSeries(z,'blue')
vs.scatterTimeSeries(x,'green')

#%% Test Kalman smoother
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
