#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Just a test of BBVI for a super simple static case
z = v1 ---latent variable
x = Bz+v2 - observed variable

v1 ~ N(0,s1^2) 
v2 ~ N(0, s2^2)

so 

z ~ p(z) = N(0,s1)
x ~ p(x|z) = N(Bz, s2)
"""

#import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#%%
'''*******************FUNCTIONS***************'''

def evalGradLogJoint(xMeasured,s1,s2,B,muQ, t, eps):
    muVec = - (muQ + np.exp(t)*eps)/(s1*s1)
    for x in xMeasured:
        muVec += B*(x - 
                  B*(muQ + np.exp(t)*eps))/(s2*s2) 
    
    tVec = muVec*eps
    
    return np.array([muVec, tVec])
    

'''returns a numpy.array with derivative wrt mu in the first element
and derivative wrt t = log(s) in the second element'''
 
def evalGradLogQ(t,eps):
    muVec = eps*np.exp(-t)
    sVec = -1 + eps**2
    
    return np.array([muVec, sVec])

def evalEntropy(t,n):#t is log sigmaQ and we want to return 1/sigmaQ
    sVec = 1/(np.exp(t))
    muVec = 0
    return np.array([muVec, sVec])

def returnTruePosteriorVariance(s1,s2,B):
    return 1/(1/(s1*s1)+B*B/(s2*s2))

def returnTruePosteriorExpectation(mu_post2,s2,B,x):
    return mu_post2*(B*x/(s2*s2))
    

'''initialize variables'''
s1 = 1
s2 = 1
mu1 = 4
mu2 = 0
B = 5

nMeasured = 1
Bmat = B*np.identity(nMeasured)
s1Mat = s1*np.identity(nMeasured)
s2Mat = s2*np.identity(nMeasured)

'''generate our values x from this model'''
#xMeasured = np.random.randn() + B*np.random.randn() #we sampled from N(Bz, s2)
zReal = np.random.normal(mu1, s1,1)
xMeasured = np.random.normal(B*(mu2 + zReal), s2, nMeasured)
#initial guess for q(z) ~ N(mu, s^2)
mu0 = 30
s0 = 10
t0 = np.log(s0)
ld = np.array([mu0, t0])

diff = 1
n = 100 #number of samples drawn from q for each step in optimization 
t = 1
ldAll = ld.reshape(-1,ld.shape[0])
while diff > 0.00005:
    print("------------------------------------------------------------")
    print(t)
    print(ld)
    #zSamples = np.random.normal(ld[0], np.exp(ld[1]), n)
    eps = np.random.normal(0,1,n)
  
    dir = np.zeros([2])
  
    for e in eps:
        #dir = dir - evalGradLogQ(ld[1], e) + evalGradLogJoint(xMeasured,s1,s2,B,ld[0],ld[1],e)
        dir = dir + evalEntropy(ld[1], nMeasured) + evalGradLogJoint(xMeasured,s1,
                               s2,B,ld[0],ld[1],e)

    dir = (1/n)*dir
    s = "direction: " + repr(dir)
    print(s)
    
    rho = 0.003*np.power(t,-0.6)

    s = "rho: " + repr(rho)
    print(s)
    ld = ld + rho*dir
    ldAll = np.concatenate((ldAll,ld.reshape(-1,ld.shape[0])),axis=0)
    print("----------ld:")
    print(ld)
    
    t += 1
    diff = np.linalg.norm(rho*dir)

muFinished = ld[0]
sFinished = np.exp(ld[1])
s = "expected: " + repr(muFinished) + "  stdev: " + repr(sFinished)
print(s)

'''
Calculate true posterior and variance
'''
vPost = np.zeros(np.size(xMeasured))
muPost = np.zeros(np.size(xMeasured))
vPost[0] = returnTruePosteriorVariance(s1,s2,B)
muPost[0] = returnTruePosteriorExpectation(vPost[0],s2,B,np.mean(xMeasured))

for i in range(1,np.size(xMeasured)):
    vPost_temp = returnTruePosteriorVariance(s1,s2,B)
    muPost_temp = returnTruePosteriorExpectation(vPost_temp,s2,B,np.mean(xMeasured))
    
    vPost[i] = vPost[i-1]*vPost_temp/(vPost[i-1]+vPost_temp)
    muPost[i] = (muPost[i-1]*vPost_temp+muPost_temp*vPost[i-1])/(
            vPost_temp+vPost[i-1])
    
postExp = muPost[np.size(xMeasured)-1]
postVar = np.sqrt(vPost[np.size(xMeasured)-1])

s = "true expected: " + repr(postExp) + " true stdev  " + repr(postVar)
print(s)


#%% Calculate ELBO
from sympy import Symbol,lambdify,powsimp,log,exp,integrate,oo
from sympy.stats import Normal,density 

zSym = Symbol('z')
xSym = Symbol('x')
Z = Normal('Z',0,s1) # p(z)~N(0,S1^2)
pz = density(Z)(zSym)
X = Normal('X',B*zSym,s2)
pxz = density(X)(xSym)
pxDataz = pxz.subs(xSym,xMeasured[0])
n = 2
for x in xMeasured[1:]:
    pxDataz = pxDataz*pxz.subs(xSym,x)
    if n%100 == 0:
        print('{0}: Still running.'.format(n))
    n += 1
muSym = Symbol('mu')
sigmaSym = Symbol('sigma',positive = True)
Zq = Normal('Zq',muSym,sigmaSym)
qz = density(Zq)(zSym)
term1 = log(pxDataz*pz).expand(force=True)
term2 = integrate(term1*qz,(zSym,-oo,oo))
term3 = log(qz).expand(force = True)
term4 = integrate(-term3*qz,(zSym,-oo,oo))
L = term2+term4
L = L.simplify()
calL = lambdify((muSym,sigmaSym),L,'numpy') #L is ELBOW
tSym = Symbol('t')
Lt = L.subs(sigmaSym,exp(tSym)) # sigma = exp(t) t = ld[1]
Lt = Lt.expand(force=True)
calLt = lambdify((muSym,tSym),Lt,'numpy')

#%%


ldAll = np.concatenate((ldAll,np.exp(ldAll[:,1]).reshape(-1,1)),axis = 1)


delta = 0.01
x = np.arange(np.min(ldAll[:,0])-np.max(ldAll[:,0]),np.max(ldAll[:,0]), delta)
y = np.arange(np.min(ldAll[:,2]),np.max(ldAll[0,2]), delta)
X, Y = np.meshgrid(x, y)
Z = calL(X,Y)
#Z = calLt(X,Y)

plt.rcParams.update({'axes.titlesize': 'large'})
plt.figure(figsize = (7,5))
CS = plt.contour(X, Y, Z,label='contour')
plt.clabel(CS, inline=1, fontsize=10)
manual_locations = [(10,0.1)]
plt.clabel(CS, inline=1, fontsize=10,manual=manual_locations)
plt.plot(ldAll[:,0],ldAll[:,2],label='Trace of BBVI')
plt.plot(postExp, postVar, marker='o', markersize=3, color="red",
         label='Analytically calculated posterior')
plt.legend(loc='best',prop={'size':7})
plt.xlabel("$\mu$", fontsize =12)
plt.ylabel("$\sigma$", fontsize=12)
plt.savefig('TraceOfBBVI.eps',format = 'eps', dpi=300)

