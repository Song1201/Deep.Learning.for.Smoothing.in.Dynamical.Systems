# Some testing code for this project.

#%% Test Generative Model linear Gaussian function
import GenerativeModel as gm
import Visualization as vs

def testGenerativeModelLG(cZ,cX,meanZ,stdZ,meanX,stdX,numTimeSteps,numSeries,
dataFileName):
  gm.linearGaussian(cZ,cX,meanZ,stdZ,meanX,stdX,numTimeSteps,numSeries)
  z,x = gm.loadData(dataFileName)
  vs.scatterTimeSeries(z,'blue')
  vs.scatterTimeSeries(x,'green')

testGenerativeModelLG(1,2,0,0.3,0,0.1,100,20,
'Generated.Data/LG.1.2.0.0.3.0.0.1')

#%% Test Generative Model nonlinear Gaussian function
import GenerativeModel as gm
import Visualization as vs

def testGenerativeModelNLG(meanZ,stdZ,meanX,stdX,numTimeSteps,numSeries,
dataFileName):
  gm.nonlinearGaussian(meanZ,stdZ,meanX,stdX,numTimeSteps,numSeries)
  z,x = gm.loadData(dataFileName)
  vs.scatterTimeSeries(z,'blue')
  vs.scatterTimeSeries(x,'green')

testGenerativeModelNLG(0,0.3,0,0.1,100,20,'Generated.Data/NLG.0.0.3.0.0.1')

#%% Test Generative Model linear non-Gaussian function
import GenerativeModel as gm
import Visualization as vs

def testGenerativeModelLNG(cZ,cX,locZ,scaleZ,shapeX,scaleX,numTimeSteps,
  numSeries,dataFileName):
  gm.linearNonGaussian(cZ,cX,locZ,scaleZ,shapeX,scaleX,numTimeSteps,numSeries)
  z,x = gm.loadData(dataFileName)
  vs.scatterTimeSeries(z,'blue')
  vs.scatterTimeSeries(x,'green')

testGenerativeModelLNG(1,2,0,1.2,1,1,100,20,
'Generated.Data/LNG.1.2.0.1.2.1.1')

#%% Test Generative Model non-linear non-Gaussian function
import GenerativeModel as gm
import Visualization as vs

def testGenerativeModelNLNG(locZ,scaleZ,shapeX,scaleX,numTimeSteps,
  numSeries,dataFileName):
  gm.nonlinearNonGaussian(locZ,scaleZ,shapeX,scaleX,numTimeSteps,numSeries)
  z,x = gm.loadData(dataFileName)
  vs.scatterTimeSeries(z,'blue')
  vs.scatterTimeSeries(x,'green')

testGenerativeModelNLNG(0,1.2,1,1,100,20,'Generated.Data/NLNG.0.1.2.1.1')

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
