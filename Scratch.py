#%%
import numpy as np

def alpha(z):
  return 0.9*np.tanh(z)*(z<1) + (1-0.9*np.tanh(z))*(z>=1)

z = np.asarray([0.5,1,3])

print(alpha(z))