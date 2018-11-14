import numpy as np
import matplotlib.pyplot as plt

def scatterTimeSeries(data,color):
  plt.figure(figsize=(10,5))
  [plt.scatter(np.arange(data.shape[1]),row,marker='o',color=color,
  s=0.3) for row in data]
  plt.show()