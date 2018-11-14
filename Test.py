#%%
import GenerativeModel as gm
import numpy as np
import matplotlib.pyplot as plt
import Visualization as vs

#%%
def testGenerativeModelLG(cZ,cX,meanZ,stdZ,meanX,stdX,numTimeSteps,numSeries,
dataFileName):
  gm.linearGaussian(cZ,cX,meanX,stdX,meanZ,stdZ,numTimeSteps,numSeries)
  z,x = gm.loadData(dataFileName)
  vs.scatterTimeSeries(z,'blue')
  vs.scatterTimeSeries(x,'green')


#%%
testGenerativeModelLG(1,2,0,0.3,0,0.1,100,20,
'Generated.Data/LG.1.2.0.0.3.0.0.1')