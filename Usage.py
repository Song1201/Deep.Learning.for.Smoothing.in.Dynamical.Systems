# This file shows how to use the models to reproduce the project. Please run 
# this file using Jupyter Notebook or other tools that can run each cell 
# separately to see its function. See cell's comments for detailed 
# description. If something weird happens, restart the Python kernel. 
# Sample Numbers (sampleNo) range from 1 to 5000


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

testHidden, testMeasure = gm.loadData('Generated.Data/LG.Test')
numTimeSteps = testHidden.shape[1]
# Parameter setted up according to GeneratedModel 
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
import UsageFunctions as uf
uf.trainCnnPointEstimator('LG')


#%% Load CNN point estimator for linear Gaussian model and test it.
import UsageFunctions as uf
uf.testCnnPointEstimator('LG',sampleNo=1234)


#%% Build and train a CNN point estimator for non-linear Gaussian model
import UsageFunctions as uf
uf.trainCnnPointEstimator('NLG')


#%% Load CNN point estimator for non-linear Gaussian model and test it.
import UsageFunctions as uf
uf.testCnnPointEstimator('NLG',sampleNo=1234)

#%% Build and train a CNN point estimator for linear non-Gaussian model
import UsageFunctions as uf
uf.trainCnnPointEstimator('LNG')


#%% Load CNN point estimator for linear non-Gaussian model and test it.
import UsageFunctions as uf
uf.testCnnPointEstimator('LNG',sampleNo=1234)


#%% Build and train a CNN point estimator for non-linear non-Gaussian model
import UsageFunctions as uf
uf.trainCnnPointEstimator('NLNG')


#%% Load CNN point estimator for non-linear non-Gaussian model and test it.
import UsageFunctions as uf
uf.testCnnPointEstimator('NLNG',sampleNo=2000)


#%% Build and train a RNN point estimator for linear Gaussian model
import UsageFunctions as uf
uf.trainRnnPointEstimator('LG')


#%% Load RNN point estimator for linear Gaussian model and test it.
import UsageFunctions as uf
uf.testRnnPointEstimator('LG',sampleNo=1234)


#%% Build and train a RNN point estimator for non-linear Gaussian model
import UsageFunctions as uf
uf.trainRnnPointEstimator('NLG')


#%% Load RNN point estimator for non-linear Gaussian model and test it.
import UsageFunctions as uf
uf.testRnnPointEstimator('NLG',sampleNo=1234)

#%% Build and train a RNN point estimator for linear non-Gaussian model
import UsageFunctions as uf
uf.trainRnnPointEstimator('LNG')


#%% Load RNN point estimator for linear non-Gaussian model and test it.
import UsageFunctions as uf
uf.testRnnPointEstimator('LNG',sampleNo=1234)


#%% Build and train a RNN point estimator for non-linear non-Gaussian model
import UsageFunctions as uf
uf.trainRnnPointEstimator('NLNG')


#%% Load RNN point estimator for non-linear non-Gaussian model and test it.
import UsageFunctions as uf
uf.testRnnPointEstimator('NLNG',sampleNo=2000)


#%%
import BBVI as vi
vi.BBVI()