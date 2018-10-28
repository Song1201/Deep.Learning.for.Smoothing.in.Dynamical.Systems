import numpy as np
from smootherKalman import smoothKalman
from GenerativeModel import LG, LNG, NLG, NLNG
#%%
xTest = np.load(r'C:\Project16\generateData\LG200xTest.dat')

dim = 1
zDim = dim
xDim = dim
z0 = np.zeros(dim)
A = 0.9*np.identity(dim)
B = 3.5*np.identity(dim)
Q = np.identity(dim)
R = Q
Q0 = Q


'''generate data'''
#%%

model = LG(zDim, xDim, A, B, Q, R, z0, Q0)
meanSmoothed = np.zeros((xTest.shape))
covSmoothed = np.zeros((xTest.shape))
for i in range(xTest.shape[0]):
    dump,meanSmoothed[i],covSmoothed[i] = smoothKalman(xTest[i], model)
    
meanSmoothed = meanSmoothed[:,:,0]
covSmoothed = covSmoothed[:,:,0]

meanSmoothed.dump('LGMeanKalman.dat')
covSmoothed.dump('LGCovKalman.dat')
