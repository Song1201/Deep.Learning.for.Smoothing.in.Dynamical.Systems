#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:14:02 2017

@author: hannes
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:14:46 2017

@author: hannes
"""

import numpy as np
    
def smoothKalman(xData, model):
    #%%
    nTimeSteps = np.shape(xData)[0]
    zDim, xDim, A, B, Q, R, z0, Q0 = model.returnParameters()
    '''
    Step 1: Compute forward estimation for all times (ie just do the normal Kalman
    filter)
    '''
    z_apr = np.zeros([nTimeSteps, zDim])
    z_post = np.zeros([nTimeSteps, zDim])
    
    z_apr[0] = z0
    #z_post[0] = z0
    
    cov_apr = np.zeros([nTimeSteps, zDim, zDim])
    cov_apr[0] = Q0
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

    '''
    Step 2: smoothing
    '''
    z_smooth = np.zeros([nTimeSteps, zDim])
    z_smooth[nTimeSteps-1] = z_post[nTimeSteps-1]
    cov_smooth = np.zeros([nTimeSteps, zDim, zDim])
    for t in range(0,nTimeSteps-1):
        now = nTimeSteps-2-t
        last = nTimeSteps-1-t #this is actually the future but whatevs
        C = np.matmul(np.matmul(cov_post[now],np.transpose(A)),
                      np.linalg.inv(cov_apr[last]))
        z_smooth[now] = z_post[now] + np.matmul(C,(z_smooth[last]-z_apr[last]))
        cov_smooth[now] = cov_post[now] + np.matmul(np.matmul(C, 
                  (cov_smooth[last]-cov_apr[last])),np.transpose(C))
    
# =============================================================================
#     for t in reversed(range(0,nTimeSteps-1)):
#         print(t)
#         m_k1 = np.matmul(A, z_post[t])
#         P_k1 = np.matmul(np.matmul(A,cov_post[t+1]),np.transpose(A)) + Q
#         C = np.matmul(np.matmul(cov_apr[t],A),np.linalg.inv(P_k1))
#         z_smooth[t] = z_post[t] + np.matmul(C,(z_smooth[t+1]-m_k1))
#         cov_smooth[t] = cov_post[t] + np.matmul(np.matmul(C, (cov_smooth[t+1] - 
#                   P_k1)), np.transpose(C))
# =============================================================================

    tList = list(range(0,nTimeSteps))
    cov_smooth = np.reshape(cov_smooth, [nTimeSteps, 1])
    
    return tList, z_smooth, cov_smooth
    
