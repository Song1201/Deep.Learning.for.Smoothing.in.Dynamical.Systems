#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
"""
Class with generative models
"""
class GenerativeModel(object):
    
    def __init__(self, zDim, xDim):
        self.zDim = zDim
        self.xDim = xDim
        
    def evaluateLogObs(self):
        '''
        should evaluate the log of the joint PDF for the model
        given som inputs
        '''
        raise Exception('Method not implemented')
        
    def generateSamples(self):
        '''
        should generate samples from the class, latent states z
        and observed states
        '''
        raise Exception('Method not implemented')
        
    def returnParameters(self):
        '''
        should return the parameters of the current model
        '''
        raise Exception('Method not implemented')
        
    def __repr__(self):
        return "Generative model"
    
class LG(GenerativeModel):
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
    '''
    def __init__(self, zDim, xDim, A, B, Q, R, z0, Q0):
        
        super(LG, self).__init__(zDim, xDim)
        self.A = A
        self.Q = Q
        self.Q0 = Q0
        self.z0 = z0
        self.B = B
        self.R = R
    def evaluateLogObs(self):
        raise Exception("method evaluateLogJoint is not implemented yet")
        
        
    '''
    Generates samples from generative model self from for T timesteps
    from t = 0 to t = nTimeSteps -1
    '''
    def generateSamples(self, nTimeSteps):
        z = np.zeros([nTimeSteps, self.zDim])
        #z[0] = np.random.multivariate_normal(self.z0,self.Q0)
        #z[0] = 0
    
        x = np.zeros([nTimeSteps, self.xDim])
        x[0] = np.random.multivariate_normal(np.matmul(self.B,z[0]),self.R)
        if nTimeSteps == 0:
            return z,x
        '''
        Making the time series
        '''
        for i in range(1,nTimeSteps): 
            z[i] = np.random.multivariate_normal(np.matmul(self.A,z[i-1]), self.Q)
            x[i] = np.random.multivariate_normal(np.matmul(self.B,z[i]), self.R) 
        
        return z, x
    
    def returnParameters(self):
        return self.zDim, self.xDim, self.A, self.B, self.Q, self.R, self.z0, self.Q0
    

class NLG(GenerativeModel):
    '''
    z[t+1] = f(z[t]) + v1 ---latent variable
    x[t] = Bz[t] + v2 - observed variable
    f(z[t])=Asin(z[t])
    
    and
    z[0] = z0 + v0 
    
    v0 ~ N(0,Q0)
    v1 ~ N(0,Q) 
    v2 ~ N(0, R)
    v0,v1,v2 are uncorrelated (or?)
    
    so 
    p(z[0]) = N(z0, Q0)
    p(z[t+1]|z[t]) = N(f(z[t]), Q)
    p(x[t]|z[t]) = N(Bz[t], R)
    
    so the parameters that completely define the model are:
        A, B, Q, R, z0, Q0 (and the dimensions of z and x)
    '''
    def __init__(self, zDim, xDim, A, B, Q, R, z0, Q0):
        
        super(NLG, self).__init__(zDim, xDim)
        self.A = A
        self.Q = Q
        self.Q0 = Q0
        self.z0 = z0
        self.B = B
        self.R = R
    def evaluateLogObs(self):
        raise Exception("method evaluateLogJoint is not implemented yet")
        
    '''
    Generates samples from generative model self from for T timesteps
    from t = 0 to t = nTimeSteps -1
    '''
    '''
    def generateSamples(self, nTimeSteps):
        z = np.zeros([nTimeSteps, self.zDim])
        z[0] = np.random.multivariate_normal(self.z0,self.Q0)

        
        x = np.zeros([nTimeSteps, self.xDim])
        x[0] = np.random.multivariate_normal(np.matmul(self.B,z[0]),self.R)
        if nTimeSteps == 0:
            return z,x
        
        #Making the time series
        
        for i in range(1,nTimeSteps):
            z[i] = np.random.multivariate_normal(np.matmul(self.A,np.sin(100*z[i-1])), self.Q)
            x[i] = np.random.multivariate_normal(z[i]*z[i], self.R) 
        
        return z, x
    '''
    def generateSamples(self, nTimeSteps):
        z = np.zeros([nTimeSteps, self.zDim])
        #z[0] = np.random.multivariate_normal(self.z0,self.Q0)

        
        x = np.zeros([nTimeSteps, self.xDim])
        x[0] = np.random.multivariate_normal(np.matmul(self.B,z[0]),self.R)
        if nTimeSteps == 0:
            return z,x
        '''
        Making the time series
        '''
        for i in range(1,nTimeSteps):
# =============================================================================
#             if z[i-1] < 1:
#                 term = np.matmul(self.A,np.tanh(z[i-1]))
#             else:
#                 term = 1 - np.matmul(self.A,np.tanh(z[i-1]))
#             z[i] = np.random.multivariate_normal(term, self.Q)
# =============================================================================
            z[i] = np.random.multivariate_normal(np.matmul(self.A,
             np.sin(100*z[i-1])), self.Q)

            x[i] = np.random.multivariate_normal(z[i]*z[i]*z[i], self.R) 
        
        return z, x
    
    def returnParameters(self):
        return self.zDim, self.xDim, self.A, self.B, self.Q, self.R, self.z0, self.Q0
    
class LNG(GenerativeModel):
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
    '''
    def __init__(self, zDim, xDim, A, B, Q, R, z0, Q0):
        
        super(LNG, self).__init__(zDim, xDim)
        self.A = A
        self.Q = Q
        self.Q0 = Q0
        self.z0 = z0
        self.B = B
        self.R = R
    def evaluateLogObs(self):
        raise Exception("method evaluateLogJoint is not implemented yet")
        
        
    '''
    Generates samples from generative model self from for T timesteps
    from t = 0 to t = nTimeSteps -1
    '''
    def generateSamples(self, nTimeSteps):
        z = np.zeros([nTimeSteps, self.zDim])
        #z[0] = 0
    
        x = np.zeros([nTimeSteps, self.xDim])
        x[0] = np.matmul(self.B, z[0]) + np.random.gamma([1])
        if nTimeSteps == 0:
            return z,x
        '''
        Making the time series
        '''
        for i in range(1,nTimeSteps): 
#            z[i] = np.random.multivariate_normal(np.matmul(self.A,z[i-1]), self.Q)
#            x[i] = np.random.multivariate_normal(np.matmul(self.B,z[i]), self.R) 
            z[i] = np.matmul(self.A,z[i-1]) + np.random.logistic(loc = 0.0, scale = 1.2)
            x[i] = np.matmul(self.B,z[i]) + np.random.gamma([1])
        
        return z, x
    
    def returnParameters(self):
        return self.zDim, self.xDim, self.A, self.B, self.Q, self.R, self.z0, self.Q0
    
class NLNG(GenerativeModel):
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
    '''
    def __init__(self, zDim, xDim, A, B, Q, R, z0, Q0):
        
        super(NLNG, self).__init__(zDim, xDim)
        self.A = A
        self.Q = Q
        self.Q0 = Q0
        self.z0 = z0
        self.B = B
        self.R = R
    def evaluateLogObs(self):
        raise Exception("method evaluateLogJoint is not implemented yet")
        
        
    '''
    Generates samples from generative model self from for T timesteps
    from t = 0 to t = nTimeSteps -1
    '''
    def generateSamples(self, nTimeSteps):
        z = np.zeros([nTimeSteps, self.zDim])
        #z[0] = 0
    
        x = np.zeros([nTimeSteps, self.xDim])
        x[0] = np.matmul(self.B, z[0]) + np.random.gamma([1])
        if nTimeSteps == 0:
            return z,x
        '''
        Making the time series
        '''
        for i in range(1,nTimeSteps): 
#            z[i] = np.random.multivariate_normal(np.matmul(self.A,z[i-1]), self.Q)
#            x[i] = np.random.multivariate_normal(np.matmul(self.B,z[i]), self.R)
# =============================================================================
#             if z[i-1] < 1:
#                 term = np.matmul(self.A,np.tanh(z[i-1]))
#             else:
#                 term = 1 - np.matmul(self.A,np.tanh(z[i-1]))            
# =============================================================================

            z[i] = np.matmul(self.A,np.sin(100*z[i-1])) + np.random.logistic(loc = 0.0, scale = 1.2)
            #z[i] = term + np.random.logistic(loc = 0.0, scale = 1.2)            
            x[i] = z[i]*z[i]*z[i] + np.random.gamma([1])
        
        return z, x
    
    def returnParameters(self):
        return self.zDim, self.xDim, self.A, self.B, self.Q, self.R, self.z0, self.Q0
    
