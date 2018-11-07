import numpy as np
import matplotlib.pyplot as plt
#import scipy as

# generate data points from the posterior distribution
mp = np.array([0,1]) # mean
sp = np.array([1,2]) # standard variance
rho = 0.5 # the correlation between each 1-d gaussian distribution
cov = np.array([[sp[0]**2,rho*sp[0]*sp[1]],[rho*sp[0]*sp[1],sp[1]**2]])
z = np.random.multivariate_normal(mp,cov,5000)
plt.figure(figsize=(10,10))
plt.scatter(z[:,0],z[:,1],s = 10)
plt.show()

#mu = np.random.multivariate_normal(mp,cov)

#z = np.random.normal(0,1,10) 
V1 = np.random.normal(loc=0,scale=1,size=1000)
V2 = np.random.normal(loc=1,scale=2,size=1000)

T = 1000  