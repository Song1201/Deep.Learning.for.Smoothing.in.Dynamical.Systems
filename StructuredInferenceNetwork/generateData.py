import numpy as np
from GenerativeModel import *
#%%

dim = 1
zDim = dim #latent variable
xDim = dim #observed variable
z0 = np.zeros(dim)
A = 0.9*np.identity(dim)
B = 3.5*np.identity(dim)
Q = np.identity(dim)
R = Q
Q0 = Q


model = NLG(zDim, xDim, A, B, Q, R, z0, Q0)

nTimeSteps = 25
N = 5000 # the number of samples
Zdata = np.empty((N,nTimeSteps,zDim),dtype=np.float64)
Xdata = np.empty((N,nTimeSteps,xDim),dtype=np.float64)
# generate data and restore them in array
for n in range(N):
    Zdata[n],Xdata[n] = model.generateSamples(nTimeSteps)
    if n%5000 == 0:
        print(n,' running')


#%% 
Zdata.dump('zTrainData1.dat')
Xdata.dump('xTrainData1.dat')

