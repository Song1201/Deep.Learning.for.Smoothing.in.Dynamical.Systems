import numpy as np

def loadData(fileName):
  data = np.load(fileName)
  numSeries = (int)(data.shape[0]/2)
  z = data[:numSeries,:]
  x = data[numSeries:,:]
  return z,x

def linearGaussian(cZ,cX,meanZ,stdZ,meanX,stdX,numTimeSteps,numSeries):
  z = np.zeros((numSeries,numTimeSteps))
  x = np.zeros((numSeries,numTimeSteps))
  
  z[:,0] = np.random.normal(meanZ,stdZ,(numSeries))
  x[:,0] = cX*z[:,0]+np.random.normal(meanX,stdX,(numSeries))
  for i in range(1,numTimeSteps):
    z[:,i] = cZ*z[:,i-1]+np.random.normal(meanZ,stdZ,(numSeries))
    x[:,i] = cX*z[:,i-1]+np.random.normal(meanX,stdX,(numSeries))

  np.append(z,x,0).dump('Generated.Data/LG.'+str(cZ)+'.'+str(cX)+'.'+
  str(meanZ)+'.'+str(stdZ)+'.'+str(meanX)+'.'+str(stdX))
  
