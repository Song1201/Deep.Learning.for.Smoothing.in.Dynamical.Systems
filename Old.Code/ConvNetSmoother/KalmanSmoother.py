#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:14:46 2017

@author: hannes
"""

import numpy as np
import matplotlib.pyplot as plt
from GenerativeModel import LG
from GenerativeModel import NLG


dim = 1
zDim = dim #latent variable
xDim = dim #observed variable
z0 = np.zeros(dim)
A = 0.9*np.identity(dim)
B = -A
Q = np.identity(dim)
R = Q
Q0 = Q

#%%
model = LG(zDim, xDim, A, B, Q, R, z0, Q0)

nTimeSteps = 200
zData, xData = model.generateSamples(nTimeSteps)

#%%
'''
 z[t+1] = Az[t] + v1 ---latent variable
    x[t] = Bz[t] + v2 - observed variable
    
    and
    z[0] = z0 + v0 
    
    v0 ~ N(0,Q0)
    v1 ~ N(0,Q) 
    v2 ~ N(0, R)
    v0,v1,v2 are uncorrelated (or?)
    
    so 
    p(z[0]) = N(z0, Q0)
    p(z[t+1]|z[t]) = N(Az[t], Q)
    p(x[t]|z[t]) = N(Bz[t], R)
    
    so the parameters that completely define the model are:
        A, B, Q, R, z0, Q0 (and the dimensions of z and x)

Step 1: Compute forward estimation for all times (ie just do the normal Kalman
filter)


'''
z_apr = np.zeros([nTimeSteps, zDim])
z_post = np.zeros([nTimeSteps, zDim])

z_apr[0] = z0
z_post[0] = z0

cov_apr = np.zeros([nTimeSteps, zDim, zDim])

cov_post = np.zeros([nTimeSteps, zDim, zDim])
cov_post[0] = Q0
for t in range(0,nTimeSteps-1):
    z_apr[t+1] = np.matmul(A,z_post[t])
    cov_apr[t+1] = np.matmul(np.matmul(A,cov_post[t]),np.transpose(A)) + Q
    
    innov = xData[t+1] - np.matmul(B,z_apr[t+1])
    S = R + np.matmul(np.matmul(B,cov_apr[t+1]),np.transpose(B)) #innovation covariance
    K = np.matmul(np.matmul(cov_apr[t+1],np.transpose(B)),
                  np.linalg.inv(S))
    z_post[t+1]  = z_apr[t+1] + np.matmul(K,innov)
    cov_post[t+1] = cov_apr[t+1] - np.matmul(np.matmul(K,S),
            np.transpose(K))
    print(K)
    
    
    
'''
Kalman smoother - the wikipedia way
kom ihåg att vid t = Tfinal är z_smooth[t] = z_apr[t]
nu = t-1
'''
z_smooth = np.zeros([nTimeSteps, zDim])
z_smooth[nTimeSteps-1] = z_post[nTimeSteps-1]
cov_smooth = np.zeros([nTimeSteps, zDim, zDim])
cov_smooth[nTimeSteps-1] = cov_post[nTimeSteps-1]
for t in reversed(range(0,nTimeSteps-1)):
    now = nTimeSteps-2-t
    last = nTimeSteps-1-t 
    C = np.matmul(np.matmul(cov_post[t],np.transpose(A)),np.linalg.inv(cov_apr[t+1]))
    z_smooth[t] = z_post[t] + np.matmul(C,(z_smooth[t+1]-z_apr[t+1]))
    cov_smooth[t] = cov_post[t] + np.matmul(np.matmul(C, 
              (cov_smooth[t+1]-cov_apr[t+1])),np.transpose(C))

# =============================================================================
# for t in reversed(range(0,nTimeSteps-1)):
#     print(t)
#     m_k1 = np.matmul(A, z_post[t])
#     P_k1 = np.matmul(np.matmul(A,cov_post[t+1]),np.transpose(A)) + Q
#     C = np.matmul(np.matmul(cov_apr[t],A),np.linalg.inv(P_k1))
#     z_smooth[t] = z_post[t] + np.matmul(C,(z_smooth[t+1]-m_k1))
#     cov_smooth[t] = cov_post[t] + np.matmul(np.matmul(C, (cov_smooth[t+1] - 
#               P_k1)), np.transpose(C))
# =============================================================================

tList = list(range(0,nTimeSteps))
#plt.figure(figsize=(20,10))
plt.scatter(tList, zData, color = 'black', label = 'true z', s = 5)
#plt.plot(tList, xData, color = 'green')
plt.plot(tList,z_post,color='red', label = 'kalman filter')
plt.plot(tList,z_smooth, color = 'blue', label = 'kalman smoother')
plt.legend()

