import numpy as np

def loadData(fileName):
  data = np.load(fileName)
  numSeries = (int)(data.shape[0]/2)
  z = data[:numSeries,:]
  x = data[numSeries:,:]
  return z,x

# z[t+1] = cZ*z[t] + N(meanZ,stdZ) hidden variable
# x[t] = cX*z[t] + N(meanX,stdX) measure variable
# z[0] = N(meanZ,stdZ)
def linearGaussian(cZ,cX,meanZ,stdZ,meanX,stdX,numTimeSteps,numSeries):
  z = np.zeros((numSeries,numTimeSteps))
  x = np.zeros((numSeries,numTimeSteps))
  
  z[:,0] = np.random.normal(meanZ,stdZ,(numSeries))
  x[:,0] = cX*z[:,0]+np.random.normal(meanX,stdX,(numSeries))
  for i in range(1,numTimeSteps):
    z[:,i] = cZ*z[:,i-1]+np.random.normal(meanZ,stdZ,(numSeries))
    x[:,i] = cX*z[:,i]+np.random.normal(meanX,stdX,(numSeries))

  np.append(z,x,0).dump('Generated.Data/LG.'+str(cZ)+'.'+str(cX)+'.'+
  str(meanZ)+'.'+str(stdZ)+'.'+str(meanX)+'.'+str(stdX))
  
# z[t+1] = alpha(z[t]) + N(meanZ,stdZ) hidden variable
# x[t] = beta(z[t]) + N(meanX,stdX) measure variable
# z[0] = N(meanZ,stdZ)
def nonlinearGaussian(meanZ,stdZ,meanX,stdX,numTimeSteps,numSeries):
  z = np.zeros((numSeries,numTimeSteps))
  x = np.zeros((numSeries,numTimeSteps))
  
  z[:,0] = np.random.normal(meanZ,stdZ,(numSeries))
  x[:,0] = beta(z[:,0])+np.random.normal(meanX,stdX,(numSeries))
  for i in range(1,numTimeSteps):
    z[:,i] = alpha(z[:,i-1])+np.random.normal(meanZ,stdZ,(numSeries))
    x[:,i] = beta(z[:,i])+np.random.normal(meanX,stdX,(numSeries))

  np.append(z,x,0).dump('Generated.Data/NLG.'+str(meanZ)+'.'+str(stdZ)+'.'+
    str(meanX)+'.'+str(stdX))

# alpha(z) = 0.9tanh(z) z<1  alpha(z) = 1-0.9tanh(z) z>=1
# z can be an array. In this case, it will typically be an 1-D array
def alpha(z):
  return 0.9*np.tanh(z)*(z<1) + (1-0.9*np.tanh(z))*(z>=1)

def beta(z):
  return np.power(z,3)

# z[t+1] = cZ*z[t] + logistic(locZ,scaleZ) hidden variable
# x[t] = cX*z[t] + gamma(shapeX,scaleX) measure variable
# z[0] = logistic(locZ,scaleZ)
def linearNonGaussian(cZ,cX,locZ,scaleZ,shapeX,scaleX,numTimeSteps,numSeries):
  z = np.zeros((numSeries,numTimeSteps))
  x = np.zeros((numSeries,numTimeSteps))
        
  z[:,0] = np.random.logistic(locZ,scaleZ,(numSeries))
  x[:,0] = cX*z[:,0] + np.random.gamma(shapeX,scaleX,(numSeries))
  for i in range(1,numTimeSteps):
    z[:,i] = cZ*z[:,i-1]+np.random.logistic(locZ,scaleZ,(numSeries))
    x[:,i] = cX*z[:,i]+np.random.gamma(shapeX,scaleX,(numSeries))

  np.append(z,x,0).dump('Generated.Data/LNG.'+str(cZ)+'.'+str(cX)+'.'+
  str(locZ)+'.'+str(scaleZ)+'.'+str(shapeX)+'.'+str(scaleX))

# z[t+1] = alpha(z[t]) + logistic(locZ,scaleZ) hidden variable
# x[t] = beta(z[t]) + gamma(shapeX,scaleX) measure variable
# z[0] = logistic(locZ,scaleZ)
def nonlinearNonGaussian(locZ,scaleZ,shapeX,scaleX,numTimeSteps,numSeries):
  z = np.zeros((numSeries,numTimeSteps))
  x = np.zeros((numSeries,numTimeSteps))
  
  z[:,0] = np.random.logistic(locZ,scaleZ,(numSeries))
  x[:,0] = beta(z[:,0]) + np.random.gamma(shapeX,scaleX,(numSeries))
  for i in range(1,numTimeSteps):
    z[:,i] = alpha(z[:,i-1])+np.random.logistic(locZ,scaleZ,(numSeries))
    x[:,i] = beta(z[:,i])+np.random.gamma(shapeX,scaleX,(numSeries))

  np.append(z,x,0).dump('Generated.Data/NLNG.'+str(locZ)+'.'+str(scaleZ)+'.'+
    str(shapeX)+'.'+str(scaleX))