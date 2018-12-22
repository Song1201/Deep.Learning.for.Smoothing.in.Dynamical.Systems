#%% A program to show how BBVI(black box variational inference) works using 
# a simple static case
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol,lambdify,powsimp,log,exp,integrate,oo
from sympy.stats import Normal,density 

# Evaluate the gradient of the log term of the joint distribution of x and z in
# Equation (10) in the report.
def evalGradLogJoint(x,stdZ,stdX,C,meanVI,logStdVI,epsilon):
  meanGrad = -(meanVI+np.exp(logStdVI)*epsilon)/(stdZ**2) + \
    C*(x-C*(meanVI+np.exp(logStdVI)*epsilon))/(stdX**2)     
  logStdGrad = meanGrad*epsilon  
  return np.array([meanGrad,logStdGrad])

# Evaluate the gradient of the entropy term of the assumed posterior 
# distribution q(z|x) in Equation (10) in the report.
def evalEntropy(logStdVI):
  return np.array([0, 1/(np.exp(logStdVI))])

def returnTruePosteriorVariance(stdZ,stdX,C):
    return 1 / (1/stdZ**2 + C**2/stdX**2)
 
def returnTruePosteriorMean(var,stdX,C,x):
    return var*C*x/stdX**2 

# Return the mean and standard deviation of the posterior distribution 
# calculated using known formula, which means this is the true value.
def calculateTruePosterior(stdZ,C,stdX,x):
  var = returnTruePosteriorVariance(stdZ,stdX,C)
  mean = returnTruePosteriorMean(var,stdX,C,x)
  return mean,np.sqrt(var)

# Create a function to calculate ELBO (Evidence Lower BOund)
def createElbo(x,stdZ,C,stdX):
  zSym = Symbol('z')
  xSym = Symbol('x')
  Z = Normal('Z',0,stdZ) # p(z)~N(0,S1^2)
  pz = density(Z)(zSym)
  X = Normal('X',C*zSym,stdX)
  pxz = density(X)(xSym)
  pxDataz = pxz.subs(xSym,x)

  muSym = Symbol('mu')
  sigmaSym = Symbol('sigma',positive = True)
  Zq = Normal('Zq',muSym,sigmaSym)
  qz = density(Zq)(zSym)
  term1 = log(pxDataz*pz).expand(force=True)
  term2 = integrate(term1*qz,(zSym,-oo,oo))
  term3 = log(qz).expand(force = True)
  term4 = integrate(-term3*qz,(zSym,-oo,oo))
  elbo = term2+term4
  elbo = elbo.simplify()
  return lambdify((muSym,sigmaSym),elbo,'numpy')

# Make sure function calculateElbo is already been created before calling this
# function
def plotBBVI(meanVI,stdVI,trueMeanPost,trueStdPost,calculateElbo,elbo):
  plt.figure(figsize = (10,5))
  plt.plot(elbo,color='blue')
  plt.show()

  delta = 0.01
  x = np.arange(np.min(meanVI)-np.max(meanVI),np.max(meanVI), delta)
  y = np.arange(np.min(stdVI),np.max(stdVI), delta)
  X, Y = np.meshgrid(x, y)
  Z = calculateElbo(X,Y)

  plt.rcParams.update({'axes.titlesize': 'large'})
  plt.figure(figsize = (10,5))
  CS = plt.contour(X, Y, Z)
  plt.clabel(CS, inline=1, fontsize=10)
  manual_locations = [(10,0.1)]
  plt.clabel(CS, inline=1, fontsize=10,manual=manual_locations)
  plt.plot(meanVI,stdVI,label='Trace of BBVI')
  plt.plot(trueMeanPost,trueStdPost, marker='o', markersize=3, color="red",
    label='Analytically calculated posterior')
  plt.legend(loc='best',prop={'size':7})
  plt.xlabel("$\mu$", fontsize =12)
  plt.ylabel("$\sigma$", fontsize=12)
  plt.show()

def BBVI():
  stdZ = 1
  stdX = 1
  meanZ = 4
  meanX = 0
  C = 5
  # z = v1 ---latent variable
  # x = C*z + v2 - observed variable (measurement)
  # v1 ~ N(meanZ,stdZ) 
  # v2 ~ N(meanX, stdX)
  # So 
  # z ~ p(z) = N(meanZ,stdZ)
  # x ~ p(x|z) = N(C*z + meanX, stdX)
  z = np.random.normal(meanZ,stdZ,1)[0]
  x = np.random.normal(C*z+meanX,stdX,1)[0]
  #initial guess for q(z) ~ N(meanVI[0],stdVI[0]) VI = Variational Inference
  meanVI = np.array([30])
  stdVI = np.array([10])
  logStdVI = np.array([np.log(10)])
  calculateElbo = createElbo(x,stdZ,C,stdX)
  elbo = calculateElbo(meanVI,stdVI)

  trueMeanPost,trueStdPost = calculateTruePosterior(stdZ,C,stdX,x)

  step = 1
  # Number of samples drawn from q(z|x) for each step in optimization to get 
  # noisy gradient
  numSamples = 100  
  numIter = 0
  endingStep = 0.0000005

  while step > endingStep:
    numIter += 1
    epsilons = np.random.normal(0,1,numSamples)
    gradients = np.zeros([2])
    for epsilon in epsilons:
      gradients += evalEntropy(logStdVI[-1]) + evalGradLogJoint(x,stdZ,stdX,C,
        meanVI[-1],logStdVI[-1],epsilon)  
    
    gradients /= numSamples
    stepSize = 0.003*np.power(numIter,-0.6)
    meanVI = np.append(meanVI,meanVI[-1]+stepSize*gradients[0])
    logStdVI = np.append(logStdVI,logStdVI[-1]+stepSize*gradients[1])
    stdVI = np.append(stdVI,np.exp(logStdVI[-1]))
    elbo = np.append(elbo,calculateElbo(meanVI[-1],stdVI[-1]))
    if numIter%100 == 0: 
      print("------------------------------------------------------------")
      print('Iteration: ',numIter,'Step: ',step, 'Ending Step: ',endingStep)
      plotBBVI(meanVI,stdVI,trueMeanPost,trueStdPost,calculateElbo,elbo)
      
    step = np.linalg.norm(stepSize*gradients)
    

  print("------------------------------------------------------------")
  print('Iteration: ',numIter,'Step: ',step, 'Ending Step: ',endingStep)
  plotBBVI(meanVI,stdVI,trueMeanPost,trueStdPost,calculateElbo,elbo)
  print('BBVI: ',meanVI[-1],stdVI[-1])
  print('True: ',trueMeanPost,trueStdPost)