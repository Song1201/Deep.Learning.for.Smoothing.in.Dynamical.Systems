import numpy as np
from sympy import Symbol,lambdify,powsimp,log,exp
from sympy.stats import Normal,density
import sympy.functions as symFunc
from sympy.physics.vector import ReferenceFrame,gradient
#%%
# generative model 
zData = 10 # True zData
xData = 5*zData + np.random.normal(0,1,10) # X=5Z+V V~N(0,1), measure 50 times for zData

# build the Robbins-Monro sequence
tSeq = np.arange(1,1000)
rhoSeq = 0.001*tSeq**(-0.99) 

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
jSym = Symbol('j')
logqz = logqz.subs(sigma,exp(jSym))
dlogqzmu = logqz.diff(mu)
#dlogqzsigma = logqz.diff(sigma)
dlogqzj = logqz.diff(jSym)
# write expression for p(xData\z[s]), knew:X=5Z+V V~N(0,1)
X = Normal('X',5*zSym,1)
xSym = Symbol('x')
pxz = density(X)(xSym)
#logpxDataz = log(pxz.subs(xSym,xData[0]))
pxDataz = pxz.subs(xSym,xData[0])
n = 2
for x in xData[1:]:
#    logpxDataz = logpxDataz+log(pxz.subs(xSym,x))
    pxDataz = pxDataz*pxz.subs(xSym,x)
    if n%100 == 0:
        print('{0}: Still running.'.format(n))
    n += 1
    
# Write expression for the public factor which is used for updating both mean and std  
#pubFact = log(pz)+logpxDataz-logqz # public factor used for update both mean and std
pubFact = log(pz*pxDataz)-logqz
# Write expression for what are used for update mean and std
exprMean = dlogqzmu*pubFact
#exprStd = dlogqzsigma*pubFact
exprJ = dlogqzj*pubFact
exprMean = exprMean.expand(force=True)
#exprStd = exprStd.expand(force=True)
exprJ = exprJ.expand(force=True)
# make exprMean,exprstd become function with respect to z,mean,std
#calMeanTerm = lambdify((zSym,mu,sigma),exprMean,'numpy')
#calStdTerm = lambdify((zSym,mu,sigma),exprStd,'numpy')
calMeanTerm = lambdify((zSym,mu,jSym),exprMean,'numpy')
calJTerm = lambdify((zSym,mu,jSym),exprJ,'numpy')

    
# set the number of samples
S = 100

# Set initial parameters for q(z)
mean = 100
#std = 2
j = 1
std = np.exp(j)

# loop the algrithm
n = 1 
for rho in rhoSeq:
    z = np.random.normal(mean,std,S) # sample z for S times
    meanArr = mean*np.ones((S)) # mean array use for vectorization
#    stdArr = std*np.ones((S)) # std array use for vectorization
    jArr = j*np.ones((S))
#    meanTerm = calMeanTerm(z,meanArr,stdArr)
    meanTerm = calMeanTerm(z,meanArr,jArr)
#    print(meanTerm)
    deltaMean = rho*np.mean(meanTerm)
    mean = mean + deltaMean
#    stdTerm = calStdTerm(z,meanArr,stdArr)
    jTerm = calJTerm(z,meanArr,jArr)
#    deltaStd = rho*np.mean(stdTerm)
    deltaJ = rho*np.mean(jTerm)
#    if std + deltaStd < 0:
#        std = std/100
#    else:
#        std = std+deltaStd
    j= j+deltaJ
    std = np.exp(j)
#    print('Iteration:{0} mean:{1:.4f} std:{2:.4f} deltaMean:{3:.4f} deltaStd:{4:.4f}'\
#          .format(n,mean,std,deltaMean,deltaStd))
    print('Iteration:{0} mean:{1:.4f} std:{2:.4f} j:{5:.4f} deltaMean:{3:.4f} deltaJ:{4:.4f}'\
          .format(n,mean,std,deltaMean,deltaJ,j))    
#    if (abs(deltaMean)<abs(mean)/100) & (abs(deltaStd)<abs(std)/100):
#        break
    if (abs(deltaMean)<abs(mean)/100) & (abs(deltaJ)<abs(j)/100):
        break

    n += 1
    
    



#a = f(z,5*np.ones((S)),np.ones(S))

#hh = mu + sigma
#f = lambdify((mu,sigma),hh,'numpy')
#print(f(np.ones(5),np.ones(5)))



#print(dlogqzsigma)
    