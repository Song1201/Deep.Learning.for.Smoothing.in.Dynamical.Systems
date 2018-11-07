# Sympy code for LearnDMM
import numpy as np
from sympy import Symbol,lambdify,powsimp,log,exp,integrate,oo
from sympy.stats import Normal,density

#%%
mean = Symbol('mu')
std = Symbol('sigma',positive = True)
epsilon = Symbol('eps')
Z = Normal('Z',mean,std)
pz = density(Z)(zSym)
