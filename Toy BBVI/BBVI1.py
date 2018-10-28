import numpy as np
from sympy import Symbol,lambdify,powsimp,log,exp,integrate,oo
from sympy.stats import Normal,density
import sympy.functions as symFunc
from sympy.physics.vector import ReferenceFrame,gradient
import matplotlib.pyplot as plt
#%%
# generative model 
zData = 5 # True zData
xData = 5*zData + np.random.normal(0,2,50) # X=5Z+V V~N(0,1), measure 50 times for zData

# build the Robbins-Monro sequence
tSeq = np.arange(1,10000)
rhoSeq = 0.001*tSeq**(-0.97) 

# Use sympy to get the expression for update value
# write expression for p(z),prior Z~N(0,1)
zSym = Symbol('z')
Zp = Normal('Z',0,1) # Zp = Z prior
pz = density(Zp)(zSym)
# Write expression for q(z)
mu = Symbol('mu')
sigma = Symbol('sigma',positive=True)
Z = Normal('Z',mu,sigma)
qz = density(Z)(zSym)
# write expression for d(logq(z[s]))/d(mu), d(logq(z[s]))/d(mu)
logqz = log(qz)
dlogqzmu = logqz.diff(mu)
dlogqzsigma = logqz.diff(sigma)
# write expression for p(xData\z[s]), knew:X=5Z+V V~N(0,1)
X = Normal('X',5*zSym,1)
xSym = Symbol('x')
pxz = density(X)(xSym)
#logpxDataz = log(pxz.subs(xSym,xData[0]))
pxDataz = pxz.subs(xSym,xData[0])
n = 2
for x in xData[1:]:
    pxDataz = pxDataz*pxz.subs(xSym,x)
    if n%100 == 0:
        print('{0}: Still running.'.format(n))
    n += 1
    
# Write expression for the public factor which is used for updating both mean and std  
# public factor used for update both mean and std
pubFact = log(pz*pxDataz)-logqz
# Write expression for what are used for update mean and std
exprMean = dlogqzmu*pubFact
exprStd = dlogqzsigma*pubFact
exprMean = exprMean.expand(force=True)
exprStd = exprStd.expand(force=True)
# make exprMean,exprstd become function with respect to z,mean,std
calMeanTerm = lambdify((zSym,mu,sigma),exprMean,'numpy')
calStdTerm = lambdify((zSym,mu,sigma),exprStd,'numpy')


    
# set the number of samples
S = 100

# Set initial parameters for q(z)
mean = 50
std = 100


# loop the algrithm
n = 1 
gradAll = np.empty((0,2))
for rho in rhoSeq:
    z = np.random.normal(mean,std,S) # sample z for S times
    meanArr = mean*np.ones((S)) # mean array use for vectorization
    stdArr = std*np.ones((S)) # std array use for vectorization
    meanTerm = calMeanTerm(z,meanArr,stdArr)
    deltaMean = rho*np.mean(meanTerm)
    mean = mean + deltaMean
    stdTerm = calStdTerm(z,meanArr,stdArr)
    deltaStd = rho*np.mean(stdTerm)
    if std + deltaStd < 0:
        std = std/100
    else:
        std = std+deltaStd
    if n%100 == 0:
        print('Iteration:{0} mean:{1:.4f} std:{2:.4f} deltaMean:{3:.4f} deltaStd:{4:.4f}'\
          .format(n,mean,std,deltaMean,deltaStd))
    
    if (abs(deltaMean)<abs(mean)/10000) & (abs(deltaStd)<abs(std)/10000):
        break


    n += 1
    
print('Iteration:{0} mean:{1:.4f} std:{2:.4f} deltaMean:{3:.4f} deltaStd:{4:.4f}'\
  .format(n,mean,std,deltaMean,deltaStd))
    
#plt.figure(figsize=(10,10))
#plt.hist(xData)

#%%
px = integrate(pxz*pz,(zSym,-oo,oo))
pzx = pxz*pz/px
pzx = pzx.expand()
pzx = pzx.powsimp()


    