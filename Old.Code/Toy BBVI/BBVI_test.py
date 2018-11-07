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

'''*******************FUNCTIONS***************'''
'''
How should we do with the measurements??? we can do like this and
add them all, or we can call this function once for each data point in
our measurements and then just add all directions and take the mean for the
new direction. There are probably other options as well, but I just don't know
'''
def evalLogJoint(xMeasured, s1, s2, B, zSampled):
    rval = 0
    for x in xMeasured:
        rval = rval + (-np.log(2*np.pi*s1*s2) - (zSampled*zSampled/(2*s1*s1) + (x-B*zSampled)*
                       (x-B*zSampled)/(2*s2*s2)))
    return rval
  
def sampleQ(muQ, sQ):
     return sQ*np.random.randn() + muQ
 
def evalLogQ(muQ, logsQ, z):
    sQ = np.exp(logsQ)
    return -0.5*np.log(2*np.pi*sQ*sQ) - 0.5*((z-muQ)*(z-muQ)/(sQ*sQ))

'''returns a numpy.array with derivative wrt mu in the first element
and derivative wrt t = log(s) in the second element'''
 
def evalGradLogQ(muQ,t,z):
    muVec = (z-muQ)*np.exp(-2*t)
    sVec = -1 +((z-muQ)**2)*np.exp(-2*t)
    
    return np.array([muVec, sVec])

def returnTruePosteriorVariance(s1,s2,B):
    return 1/(1/(s1*s1)+B*B/(s2*s2))

def returnTruePosteriorExpectation(mu_post2,s2,B,xMean):
    return mu_post2*(B*xMean/(s2*s2))
    

'''initialize variables'''
s1 = 1
s2 = 1
B = 0.5

nMeasured = 10

'''generate our values x from this model'''
#xMeasured = np.random.randn() + B*np.random.randn() #we sampled from N(Bz, s2)
zReal = np.random.normal(0, s1,nMeasured)
xMeasured = np.random.normal(B*zReal, s2, nMeasured)
#initial guess for q(z) ~ N(mu, s^2)
mu0 = 10
s0 = 10
t0 = np.log(s0)
ld = np.array([mu0, t0])
ldVec = np.zeros([1000,2])
ldVec[0] = ld

diff = 1
n = 1000 #number of samples drawn from q for each step in optimization 
t = 1
tVec = np.zeros([1000])
print(ld)
while diff > 0.0001:
    print("------------------------------------------------------------")
    print(t)
    zSamples = np.random.normal(ld[0], np.exp(ld[1]), n)
    dir = np.zeros([2])
    tVec[t-1] = t 
  
    for z in zSamples:
        dir = dir + evalGradLogQ(ld[0], ld[1], z)*(evalLogJoint(xMeasured,
                            s1, s2, B,z)-evalLogQ(ld[0], ld[1], z))

    dir = (1/n)*dir
    s = "direction: " + repr(dir)
    #rint(s)
    
    rho = 0.01#*np.power(t,-5/7)
    '''
    while ld[1] + rho*dir[1] < 0:
        print(ld[1] + rho*dir[1])
        #rho1 = rho1/2
        print("trying to become negative")
        rho = rho/2
    '''
    #ld[0] = ld[0] + rho*dir[0]
    #ld[1] = ld[1] + rho1*dir[1]
    ld = ld + rho*dir
    ldVec[t] = ld
    print("----------ld:")
    print(ld)
    
    t += 1
    
    #diff = diff - 0.1
    #diff = np.abs(rho*dir[0]) + np.abs(rho1*dir[1])
    diff = np.linalg.norm(rho*dir)

muFinished = ld[0]
sFinished = np.exp(ld[1])
s = "mean: " + repr(muFinished) + "  standard deviation: " + repr(sFinished)
print(s)
sPost = returnTruePosteriorVariance(s1,s2,B)
muPostReal = returnTruePosteriorExpectation(sPost,s2,B,np.mean(xMeasured))
s = "(kind of) true expected: " + repr(muPostReal) + " true standard deviation  " + repr(np.sqrt(sPost))
print(s)