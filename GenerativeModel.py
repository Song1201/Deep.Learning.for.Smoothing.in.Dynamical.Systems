import numpy as np

def loadData(fileName):
  data = np.load(fileName)
  numSeries = (int)(data.shape[0]/2)
  z = data[:numSeries,:]
  x = data[numSeries:,:]
  return z,x

# z[t] = f(z[t-1]) + v1, x[t] = g(z[t]) + v2, z[0] = z0
# v1, v2 are probability distributions of noises
def _generateData(f,g,v1,v2,z0,numTimeSteps,numSeries):
  z = np.zeros((numSeries,numTimeSteps))
  x = np.zeros((numSeries,numTimeSteps))
  
  z[:,0] = z0
  x[:,0] = g(z[:,0]) + v2(numSeries)
  for i in range(1,numTimeSteps):
    z[:,i] = f(z[:,i-1]) + v1(numSeries)
    x[:,i] = g(z[:,i]) + v2(numSeries)
  
  return z,x

# Functions and noise probability distributions for generative model
_fLinear = lambda z: 0.9*z
_gLinear = lambda z: 3.5*z
_v1Gaussian = _v2Gaussian = lambda numSeries: np.random.normal(0,1,(numSeries))
_fNonLinear = lambda z: 0.9*np.tanh(z)*(z<1) + (1-0.9*np.tanh(z))*(z>=1) 
_gNonLinear = lambda z: np.power(z,3)
_v1NonGaussian = lambda numSeries: np.random.logistic(0,1.2,(numSeries))
_v2NonGaussian = lambda numSeries: np.random.gamma(1,1,(numSeries))

def linearGaussian(numTimeSteps,numSeries):
  z,x = _generateData(_fLinear,_gLinear,_v1Gaussian,_v2Gaussian,0,numTimeSteps,
    numSeries)
  np.append(z,x,0).dump('Generated.Data/LG')

def nonLinearGaussian(numTimeSteps,numSeries):
  z,x = _generateData(_fNonLinear,_gNonLinear,_v1Gaussian,_v2Gaussian,0,
    numTimeSteps,numSeries)
  np.append(z,x,0).dump('Generated.Data/NLG')  

def linearNonGaussian(numTimeSteps,numSeries):
  z,x = _generateData(_fLinear,_gLinear,_v1NonGaussian,_v2NonGaussian,0,
    numTimeSteps,numSeries)
  np.append(z,x,0).dump('Generated.Data/LNG')

def nonLinearNonGaussian(numTimeSteps,numSeries):
  z,x = _generateData(_fNonLinear,_gNonLinear,_v1NonGaussian,_v2NonGaussian,0,
    numTimeSteps,numSeries)
  np.append(z,x,0).dump('Generated.Data/NLNG')  
