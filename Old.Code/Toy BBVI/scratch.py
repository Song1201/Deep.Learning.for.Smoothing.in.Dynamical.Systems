import numpy as np
import matplotlib.pyplot as plt

n = np.arange(1,10000000)
gamma = n**(-5/6)

#a = np.sum(gamma)
#print(a)

sumGamma = np.cumsum(gamma)
sumGammaSqure = np.cumsum(gamma**2)

fig = plt.figure(figsize=(10,10))
plt.plot(n,sumGamma,color='red')
plt.plot(n,sumGammaSqure,color='blue')
plt.show()

#fig = plt.figure(figsize=(10,10))
#sumGamma = 0
#sumGammaSqure = 0
#for n in range(1,100000):
#    sumGamma = sumGamma + n**(-0.55)
#    sumGammaSqure = sumGammaSqure + n**(2*(-0.55))
#    plt.scatter(n,sumGamma,color='red',s=5)
#    plt.scatter(n,sumGammaSqure,color='blue',s=5)
#
#plt.show()