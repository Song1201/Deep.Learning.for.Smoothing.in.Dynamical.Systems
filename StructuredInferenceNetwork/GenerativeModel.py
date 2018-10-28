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
        
    def evaluateLogJoint(self):
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

        '''
        TODO: Assert that A, B, Q and R have the right dimensions
        in relation to zDim and xDim
        
        if zDim == 1:
            self.A = np.array([A])
            self.Q = np.array([Q])
            self.Q0 = np.array([Q0])
            self.z0 = np.array([z0])
            if xDim == 1:
                self.B = np.array([B])
                self.R = np.array([R])
        else:
            self.A = np.array(A)
            self.Q = np.array(Q)
            self.Q0 = np.array(Q0)
            self.z0 = np.array(z0)
            if xDim == 1:
                self.B = np.array([B])
                self.R = np.array([R])
            else:
                self.B = np.array(B)
                self.R = np.array(R)
                '''
        self.A = A
        self.Q = Q
        self.Q0 = Q0
        self.z0 = z0
        self.B = B
        self.R = R
    def evaluateLogJoint(self):
        raise Exception("method evaluateLogJoint is not implemented yet")
        
    '''
    Generates samples from generative model self from for T timesteps
    from t = 0 to t = nTimeSteps -1
    '''
    def generateSamples(self, nTimeSteps):
        z = np.zeros([nTimeSteps, self.zDim])
#        z[0] = np.random.multivariate_normal(self.z0,self.Q0)
        z[0] = 0

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
    def evaluateLogJoint(self):
        raise Exception("method evaluateLogJoint is not implemented yet")
        
    '''
    Generates samples from generative model self from for T timesteps
    from t = 0 to t = nTimeSteps -1
    '''
    def generateSamples(self, nTimeSteps):
        z = np.zeros([nTimeSteps, self.zDim])
        z[0] = np.random.multivariate_normal(self.z0,self.Q0)

        
        x = np.zeros([nTimeSteps, self.xDim])
#        x[0] =  np.random.normal(np.matmul(self.B,z[0]), Rd, self.xDim)
        x[0] = np.random.multivariate_normal(np.matmul(self.B,z[0]),self.R)
        if nTimeSteps == 0:
            return z,x
        '''
        Making the time series
        '''
        for i in range(1,nTimeSteps):
            z[i] = np.random.multivariate_normal(np.matmul(self.A,np.sin(z[i-1])), self.Q)
            x[i] = np.random.multivariate_normal(np.matmul(self.B,z[i]), self.R) 
        
        return z, x
    
    def returnParameters(self):
        return self.zDim, self.xDim, self.A, self.B, self.Q, self.R, self.z0, self.Q0