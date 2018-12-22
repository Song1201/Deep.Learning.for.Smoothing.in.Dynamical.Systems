#%% A program to show how BBVI(black box variational inference) works using 
# a simple static case
import numpy as np
import matplotlib.pyplot as plt

# Evaluate the gradient of the log term of the joint distribution of x and z in
# Equation (10) in the report.
def evalGradLogJoint(X,stdZ,stdX,C,meanVI,logStdVI,epsilon):
  meanGrad = -(meanVI+np.exp(logStdVI)*epsilon)/(stdZ**2) + \
    C*(x-C*(meanVI+np.exp(logStdVI)*epsilon))/(stdX**2)     
  logStdGrad = meanGrad*epsilon  
  return np.array([meanGrad,logStdGrad])

# Evaluate the gradient of the entropy term of the assumed posterior 
# distribution q(z|x) in Equation (10) in the report.
def evalEntropy(logStdVI):
  return np.array([0, 1/(np.exp(logStdVI))])

def returnTruePosteriorVariance(s1,s2,B):
    return 1/(1/(s1*s1)+B*B/(s2*s2))

def returnTruePosteriorExpectation(mu_post2,s2,B,x):
    return mu_post2*(B*x/(s2*s2))
    

# z = v1 ---latent variable
# x = C*z + v2 - observed variable (measurement)

# v1 ~ N(meanZ,stdZ) 
# v2 ~ N(meanX, stdX)

# so 
# z ~ p(z) = N(meanZ,stdZ)
# x ~ p(x|z) = N(C*z + meanX, stdX)

stdZ = 1
stdX = 1
meanZ = 4
meanX = 0
C = 5

z = np.random.normal(meanZ,stdZ,1)
x = np.random.normal(C*z+meanX,stdX,1)
#initial guess for q(z) ~ N(meanVI[0],stdVI[0]) VI = Variational Inference
meanVI = np.array([30])
logStdVI = np.array([np.log(10)])

step = sys.maxsize
n = 100 #number of samples drawn from q for each step in optimization 
numIter = 1
# ldAll = ld.reshape(-1,ld.shape[0])
while step > 0.0005:

  eps = np.random.normal(0,1,n)

  dir = np.zeros([2])

  for e in eps:
      # dir = dir + evalEntropy(ld[1], 1) + evalGradLogJoint(x,stdZ,
      #                        stdX,C,ld[0],ld[1],e)
      dir = dir + evalEntropy(logStdVI[-1]) + evalGradLogJoint(x,stdZ,
                              stdX,C,meanVI[-1],logStdVI[-1],e)  

  dir = (1/n)*dir
  # s = "direction: " + repr(dir)
  # print(s)
  
  rho = 0.003*np.power(numIter,-0.6)

  # s = "rho: " + repr(rho)
  # print(s)
  # ld = ld + rho*dir
  # ldAll = np.concatenate((ldAll,ld.reshape(-1,ld.shape[0])),axis=0)
  meanVI = np.append(meanVI,meanVI[-1]+rho*dir[0])
  logStdVI = np.append(logStdVI,logStdVI[-1]+rho*dir[1])

  # print("----------ld:")
  # print(ld)
  if numIter%100 == 0: 
    print("------------------------------------------------------------")
    print(numIter)
    print(meanVI[-1],logStdVI[-1])
    print(step)
  
  numIter += 1
  step = np.linalg.norm(rho*dir)

# muFinished = ld[0]
# sFinished = np.exp(ld[1])
# s = "expected: " + repr(muFinished) + "  stdev: " + repr(sFinished)
# print(s)

'''
Calculate true posterior and variance
'''
vPost = np.zeros(np.size(x))
muPost = np.zeros(np.size(x))
vPost[0] = returnTruePosteriorVariance(stdZ,stdX,C)
muPost[0] = returnTruePosteriorExpectation(vPost[0],stdX,C,np.mean(x))

for i in range(1,np.size(x)):
    vPost_temp = returnTruePosteriorVariance(stdZ,stdX,C)
    muPost_temp = returnTruePosteriorExpectation(vPost_temp,stdX,C,np.mean(x))
    
    vPost[i] = vPost[i-1]*vPost_temp/(vPost[i-1]+vPost_temp)
    muPost[i] = (muPost[i-1]*vPost_temp+muPost_temp*vPost[i-1])/(
            vPost_temp+vPost[i-1])
    
postExp = muPost[np.size(x)-1]
postVar = np.sqrt(vPost[np.size(x)-1])

s = "true expected: " + repr(postExp) + " true stdev  " + repr(postVar)
print(s)


#%% Calculate ELBO
from sympy import Symbol,lambdify,powsimp,log,exp,integrate,oo
from sympy.stats import Normal,density 

zSym = Symbol('z')
xSym = Symbol('x')
Z = Normal('Z',0,stdZ) # p(z)~N(0,S1^2)
pz = density(Z)(zSym)
X = Normal('X',C*zSym,stdX)
pxz = density(X)(xSym)
pxDataz = pxz.subs(xSym,x[0])
n = 2
for x in x[1:]:
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


# ldAll = np.concatenate((ldAll,np.exp(ldAll[:,1]).reshape(-1,1)),axis = 1)
stdVI = np.exp(logStdVI)

delta = 0.01
# x = np.arange(np.min(ldAll[:,0])-np.max(ldAll[:,0]),np.max(ldAll[:,0]), delta)
# y = np.arange(np.min(ldAll[:,2]),np.max(ldAll[0,2]), delta)
x = np.arange(np.min(meanVI)-np.max(meanVI),np.max(meanVI), delta)
y = np.arange(np.min(stdVI),np.max(stdVI), delta)
X, Y = np.meshgrid(x, y)
Z = calL(X,Y)
#Z = calLt(X,Y)

plt.rcParams.update({'axes.titlesize': 'large'})
plt.figure(figsize = (7,5))
CS = plt.contour(X, Y, Z,label='contour')
plt.clabel(CS, inline=1, fontsize=10)
manual_locations = [(10,0.1)]
plt.clabel(CS, inline=1, fontsize=10,manual=manual_locations)
# plt.plot(ldAll[:,0],ldAll[:,2],label='Trace of BBVI')
plt.plot(meanVI,stdVI,label='Trace of BBVI')
plt.plot(postExp, postVar, marker='o', markersize=3, color="red",
         label='Analytically calculated posterior')
plt.legend(loc='best',prop={'size':7})
plt.xlabel("$\mu$", fontsize =12)
plt.ylabel("$\sigma$", fontsize=12)
