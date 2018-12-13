# Some functions for Usage.py

import GenerativeModel as gm
import NeuralNetwork as nn
import numpy as np
import matplotlib.pyplot as plt
import KalmanSmoother as ks

# modelType can be 'LG'/'NLG'/'LNG'/'NLNG'
def trainCnnPointEstimator(modelType):
  hidden,measure = gm.loadData('Generated.Data/' + modelType + '.Train')
  testHidden, testMeasure = gm.loadData('Generated.Data/' + modelType + '.Test')
  cnnPointEstimator = nn.CnnPointEstimator(hidden.shape[1])
  cnnPointEstimator.train(2e-4,100,200,measure,hidden,
    'Trained.Models/CNN.Point.Estimator.' + modelType + '.ckpt',testMeasure,
    testHidden,1213,showKalman=(modelType=='LG'))   

# modelType can be 'LG'/'NLG'/'LNG'/'NLNG'
def testCnnPointEstimator(modelType,sampleNo):
  sampleNo -= 1
  testHidden,testMeasure = gm.loadData('Generated.Data/' + modelType + '.Test')
  cnnPointEstimator = nn.CnnPointEstimator(testHidden.shape[1])
  estimated = cnnPointEstimator.infer(testMeasure[sampleNo],
    'Trained.Models/CNN.Point.Estimator.' + modelType + 
    '/CNN.Point.Estimator.' + modelType + '.ckpt')
  loss = cnnPointEstimator.computeLoss(estimated,testHidden[sampleNo],
    'Trained.Models/CNN.Point.Estimator.' + modelType + 
    '/CNN.Point.Estimator.' + modelType + '.ckpt')
  if modelType=='LG':
    testKalmanZ,dump = ks.loadResults('Results.Data/' + modelType + 
      '.Kalman.Results')
  plt.figure(figsize=(10,5))
  plt.scatter(np.arange(testHidden.shape[1]),testHidden[sampleNo],
    marker='o',color='blue',s=4)
  if modelType=='LG': plt.plot(testKalmanZ[sampleNo],color='green')
  plt.plot(estimated.flatten(),color='red')
  plt.show()

 # modelType can be 'LG'/'NLG'/'LNG'/'NLNG'
def trainRnnPointEstimator(modelType):
  lr = 3e-4
  hidden,measure = gm.loadData('Generated.Data/' + modelType + '.Train')
  testHidden, testMeasure = gm.loadData('Generated.Data/' + modelType + '.Test')
  rnnPointEstimator = nn.RnnPointEstimator(hidden.shape[1])
  rnnPointEstimator.train(lr,100,200,measure,hidden,
    'Trained.Models/RNN.Point.Estimator.' + modelType + '.ckpt',testMeasure,
    testHidden,1213,showKalman=(modelType=='LG'))  

# modelType can be 'LG'/'NLG'/'LNG'/'NLNG'
def testRnnPointEstimator(modelType,sampleNo):
  sampleNo -= 1
  testHidden,testMeasure = gm.loadData('Generated.Data/' + modelType + '.Test')
  rnnPointEstimator = nn.RnnPointEstimator(testHidden.shape[1])
  estimated = rnnPointEstimator.infer(testMeasure[sampleNo],
    'Trained.Models/RNN.Point.Estimator.' + modelType + 
    '/RNN.Point.Estimator.' + modelType + '.ckpt')
  loss = rnnPointEstimator.computeLoss(estimated,testHidden[sampleNo],
    'Trained.Models/RNN.Point.Estimator.' + modelType + 
    '/RNN.Point.Estimator.' + modelType + '.ckpt')
  if modelType=='LG':
    testKalmanZ,dump = ks.loadResults('Results.Data/' + modelType + 
      '.Kalman.Results')
  plt.figure(figsize=(10,5))
  plt.scatter(np.arange(testHidden.shape[1]),testHidden[sampleNo],
    marker='o',color='blue',s=4)
  if modelType=='LG': plt.plot(testKalmanZ[sampleNo],color='green')
  plt.plot(estimated.flatten(),color='red')
  plt.show()