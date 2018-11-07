import numpy as np

#lossFun = np.load(r'/home/hannes/Programming_projects/Project16/resultData/LGRNNpointHuber.dat')
#zTest = np.load(r'/home/hannes/Programming_projects/Project16/generateData/LNG200zTest.dat')
zTest = np.load(r'C:\Project16\generateData\NLG200zTestSin.dat')
zTest = zTest[:,:,0]
#zSmoothed = np.load(r'/home/hannes/Programming_projects/Project16/resultData/NLNGRNNsmoothed.dat')
#zSmoothed = np.load('NLGsinRNNsmoothed.dat')
zSmoothed = np.zeros(np.shape(zTest))
#zSmoothed = zSmoothed[0]

#MSE = np.mean(np.square(zTest-zSmoothed))
RMSE = np.sqrt(np.mean(np.square(zTest-zSmoothed)))

print(RMSE)
