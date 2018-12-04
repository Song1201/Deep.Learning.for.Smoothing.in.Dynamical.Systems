#%%
import numpy as np

#%%
class KalmanSmoother:

  def __init__(self,cZ,cX,meanZ,stdZ,meanX,stdX):
    self.cZ = cZ
    self.cX = cX
    self.meanZ = meanZ
    self.stdZ = stdZ
    self.meanX = meanX
    self.stdX = stdX

  def smoothStd(self,numTimeSteps):
    # The first hidden state is the initial state plus the noise. In this 
    # project, initial states are always 0. So the first hidden state in  
    # estimating is the mean of the Gaussian noise.  
    covPriori = np.zeros([numTimeSteps])
    covPriori[0] = self.stdZ
    covPost = np.zeros([numTimeSteps])
    covPost[0] = self.stdZ
    for i in range(1,numTimeSteps):
      covPriori[i] = self.cZ*covPost[i-1]*self.cZ + self.stdZ
      # innovation covariance
      innovCov = self.stdX + self.cX*covPriori[i]*self.cX
      kalmanGain = covPriori[i]*self.cX/innovCov
      covPost[i] = covPriori[i] - kalmanGain*innovCov*kalmanGain

    covSmooth = np.zeros([numTimeSteps])
    for i in range(numTimeSteps-2,-1,-1):
      covK = covPost[i]*self.cZ/covPriori[i+1]
      covSmooth[i] = covPost[i] + covK*(covSmooth[i+1]-covPriori[i+1])*covK

    # covSmooth has shape [numTimeSteps], because this Kalman smoother for 
    # linear Gaussian model infer the covariance only basing on the model 
    # parameters and the number of time steps, which means it will 
    # output the same covariance for all measurement samples.
    return np.sqrt(covSmooth)    

  # x are measurements. For this project, since all data are only 1-D. x is a 
  # 2-D matrix, with shape [numSamples,numTimeSteps].
  def smoothZ(self,x):
    numTimeSteps = x.shape[1]
    numSamples = x.shape[0]
    zPriori = np.zeros([numSamples,numTimeSteps])
    # The first hidden state is the initial state plus the noise. In this 
    # project, initial states are always 0. So the first hidden state in  
    # estimating is the mean of the Gaussian noise.  
    zPriori[:,0] = self.meanZ 
    zPost = np.zeros([numSamples,numTimeSteps])
    zPost[:,0] = self.meanZ
    covPriori = np.zeros([numTimeSteps])
    covPriori[0] = self.stdZ
    covPost = np.zeros([numTimeSteps])
    covPost[0] = self.stdZ
    for i in range(1,numTimeSteps):
      zPriori[:,i] = self.cZ*zPost[:,i-1]
      covPriori[i] = self.cZ*covPost[i-1]*self.cZ + self.stdZ
      innov = x[:,i] - self.cX*zPriori[:,i] # innovation
      # innovation covariance
      innovCov = self.stdX + self.cX*covPriori[i]*self.cX
      kalmanGain = covPriori[i]*self.cX/innovCov
      zPost[:,i] = zPriori[:,i] + kalmanGain*innov
      covPost[i] = covPriori[i] - kalmanGain*innovCov*kalmanGain

    zSmooth = np.zeros([numSamples,numTimeSteps])
    zSmooth[:,-1] = zPost[:,-1]
    for i in range(numTimeSteps-2,-1,-1):
      covK = covPost[i]*self.cZ/covPriori[i+1]
      zSmooth[:,i] = zPost[:,i] + covK*(zSmooth[:,i+1]-zPriori[:,i+1])

    # zSmooth has the same shape as x.
    return zSmooth

  # Smooth measurements data and save the results.
  def smooth(self,x,saveFile):
    zSmooth = self.smoothZ(x)
    stdSmooth = self.smoothStd(x.shape[1])
    np.concatenate((zSmooth,stdSmooth.reshape(1,-1)),axis=0).dump(saveFile)

  def loadResults(self,loadFile):
    results = np.load(loadFile)
    return results[:-2,:], results[-1]

