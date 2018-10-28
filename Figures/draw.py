import matplotlib.pyplot as plt
import numpy as np

#%% RNNKalmanSmootherLG
<<<<<<< HEAD
meanRNN = np.load(r'/home/hannes/Programming_projects/Project16/ConvNetSmoother/mu_StructuredInference_LG.dat')
stdRNN = np.load(r'/home/hannes/Programming_projects/Project16/ConvNetSmoother/sigma_StructuredInference_LG.dat')
meanKalman = np.load(r'/home/hannes/Programming_projects/Project16/StructuredInferenceNetwork/meanKalman.dat')
stdKalman = np.load(r'/home/hannes/Programming_projects/Project16/StructuredInferenceNetwork/stdKalman.dat')
#zTest = np.load(r'/home/hannes/Programming_projects/Project16/generateData/LnGaus200zTest.dat')
#meanRNN = np.load(r'C:\Project16\ConvNetSmoother\mu_StructuredInference_LG.dat')
#stdRNN = np.load(r'C:\Project16\ConvNetSmoother\sigma_StructuredInference_LG.dat')
#meanKalman = np.load(r'C:\Project16\StructuredInferenceNetwork\meanKalman.dat')
#stdKalman = np.load(r'C:\Project16\StructuredInferenceNetwork\stdKalman.dat')
zTest = np.load(r'/home/hannes/Programming_projects/Project16/generateData/LG200zTest.dat')
=======
#meanRNN = np.load(r'/home/hannes/Programming_projects/Project16/ConvNetSmoother/mu_StructuredInference_LG.dat')
#stdRNN = np.load(r'/home/hannes/Programming_projects/Project16/ConvNetSmoother/sigma_StructuredInference_LG.dat')
#meanKalman = np.load(r'/home/hannes/Programming_projects/Project16/StructuredInferenceNetwork/meanKalman.dat')
#stdKalman = np.load(r'/home/hannes/Programming_projects/Project16/StructuredInferenceNetwork/stdKalman.dat')
#zTest = np.load(r'/home/hannes/Programming_projects/Project16/generateData/LnGaus200zTest.dat')
meanRNN = np.load(r'C:\Project16\ConvNetSmoother\mu_StructuredInference_LG.dat')
stdRNN = np.load(r'C:\Project16\ConvNetSmoother\sigma_StructuredInference_LG.dat')
meanKalman = np.load(r'C:\Project16\StructuredInferenceNetwork\meanKalman.dat')
stdKalman = np.load(r'C:\Project16\StructuredInferenceNetwork\stdKalman.dat')
zTest = np.load(r'C:\Project16\generateData\LG200zTest.dat')
>>>>>>> 1d120a675b4cdef33a525d9840232c19c24be14e
zTest = zTest[0]
fig = plt.figure(figsize=(10,7))
#plt.plot(meanRNN,color='blue',label='BBVI')
plt.plot(stdKalman,color='red',label='Kalman', linewidth='2')
#plt.fill_between(np.arange(200),meanKalman-stdKalman, meanKalman+stdKalman,color='red')
#plt.plot(meanKalman-stdKalman,color='red',linewidth='0.7',label='Kalman')
#plt.plot(meanKalman+stdKalman,color='red',linewidth='0.7')
#plt.fill_between(np.arange(200),meanRNN-stdRNN, meanRNN+stdRNN,color='blue',label='BBVI')
plt.plot(stdRNN, color = 'blue', ls = '--', label = 'BBVI', linewidth=2)
#plt.scatter(np.arange(200),zTest,label='True Data',color='green')
plt.subplots_adjust(left=0.07,bottom=0.1,right=0.98,top=0.95)
<<<<<<< HEAD
plt.legend(prop={'size': 30})
plt.xlabel('Time',size=30)
plt.ylabel('Expected value',size=30)
=======
plt.legend(prop={'size':35})
plt.xlabel('Time',size=35)
plt.ylabel('Expected Value',size=35)
>>>>>>> 1d120a675b4cdef33a525d9840232c19c24be14e
plt.show()
fig.savefig('RNNKalmanSmootherLG_stdDev.eps',dpi=600)

#%% RNNmeanKalmanSmootherLG
meanRNN = np.load(r'C:\Project16\ConvNetSmoother\mu_StructuredInference_LG.dat')
stdRNN = np.load(r'C:\Project16\ConvNetSmoother\sigma_StructuredInference_LG.dat')
meanKalman = np.load(r'C:\Project16\StructuredInferenceNetwork\meanKalman.dat')
stdKalman = np.load(r'C:\Project16\StructuredInferenceNetwork\stdKalman.dat')
zTest = np.load(r'C:\Project16\generateData\LG200zTest.dat')
zTest = zTest[0]
fig = plt.figure(figsize=(10,7))
#plt.plot(meanRNN,color='blue',label='BBVI')
#plt.plot(meanKalman,color='red',label='Kalman')
#plt.fill_between(np.arange(200),meanKalman-stdKalman, meanKalman+stdKalman,color='red')
plt.plot(meanKalman,color='blue',linewidth='3',label='Kalman')
plt.plot(meanRNN,color='red',ls='--',dashes=(3,3),lw=3,label='BBVI')
plt.scatter(np.arange(200),zTest,label='True Data',color='black')
plt.subplots_adjust(left=0.11,bottom=0.12,right=0.98,top=0.95)
plt.legend(prop={'size': 20})
plt.xlabel('Time',size=25)
plt.ylabel('Expected Value',size=25)
plt.rc('xtick', labelsize=25)    
plt.rc('ytick', labelsize=25)
plt.show()
fig.savefig('RNNKalmanSmootherLG_mean.eps',dpi=600)

#%% RNNstdKalmanSmootherLG
meanRNN = np.load(r'C:\Project16\ConvNetSmoother\mu_StructuredInference_LG.dat')
stdRNN = np.load(r'C:\Project16\ConvNetSmoother\sigma_StructuredInference_LG.dat')
meanKalman = np.load(r'C:\Project16\StructuredInferenceNetwork\meanKalman.dat')
stdKalman = np.load(r'C:\Project16\StructuredInferenceNetwork\stdKalman.dat')
zTest = np.load(r'C:\Project16\generateData\LG200zTest.dat')
zTest = zTest[0]
fig = plt.figure(figsize=(10,7))
#plt.plot(meanRNN,color='blue',label='BBVI')
#plt.plot(meanKalman,color='red',label='Kalman')
#plt.fill_between(np.arange(200),meanKalman-stdKalman, meanKalman+stdKalman,color='red')
plt.plot(stdKalman,color='blue',linewidth='3',label='Kalman')
plt.plot(stdRNN,color='red',ls='--',dashes=(3,3),lw=3,label='BBVI')
plt.subplots_adjust(left=0.11,bottom=0.12,right=0.98,top=0.95)
plt.legend(prop={'size': 20})
plt.xlabel('Time',size=25)
plt.ylabel('Standard Deviation',size=25)
plt.rc('xtick', labelsize=25)    
plt.rc('ytick', labelsize=25)
plt.show()
fig.savefig('RNNKalmanSmootherLG_std.eps',dpi=600)
#%% LGKalmanConvNetRNNPoint
i = 0
kalman = np.load(r'C:\Project16\resultData\LGMeanKalman.dat')[i]
convNet = np.load(r'C:\Project16\resultData\LGConvNetSmoothed.dat')[i]
rnn = np.load(r'C:\Project16\resultData\LGRNNSmoothed.dat')[i]
zTest = np.load(r'C:\Project16\generateData\LG200zTest.dat')[i,:,0]
nTimeSteps = zTest.shape[0]

fig = plt.figure(figsize=(10,7))
plt.plot(kalman,color='blue',linewidth='2',label='Kalman')
plt.plot(convNet,color='green',ls='--',label='ConvNet Point Estimator',lw=3)
plt.plot(rnn,color='red',ls=':',label='RNN Point Estimator',lw=3)
plt.scatter(np.arange(kalman.shape[0]),zTest,label='True Data',color='black',s=14)
plt.subplots_adjust(left=0.11,bottom=0.12,right=0.98,top=0.95)
plt.legend(prop={'size': 20})
plt.xlabel('Time',size=25)
plt.ylabel('Amplitude',size=25)
plt.rc('xtick', labelsize=25)    
plt.rc('ytick', labelsize=25)
plt.show()
fig.savefig('LGKalmanConvNetRNNPoint.eps',dpi=600)

#%% LNGKalmanConvNetRNNPoint
i = 0
convNet = np.load(r'C:\Project16\resultData\LNGConvNetSmoothed.dat')[i]
rnn = np.load(r'C:\Project16\resultData\LNGRNNSmoothed.dat')[i]
zTest = np.load(r'C:\Project16\generateData\LNG200zTest.dat')[i,:,0]

fig = plt.figure(figsize=(10,7))
plt.plot(convNet,color='green',ls='-',label='ConvNet Point Estimator',lw=3)
plt.plot(rnn,color='red',ls='--',dashes=(3,3),label='RNN Point Estimator',lw=3)
plt.scatter(np.arange(kalman.shape[0]),zTest,label='True Data',color='black',s=14)
plt.subplots_adjust(left=0.11,bottom=0.12,right=0.98,top=0.95)
plt.legend(prop={'size': 20})
plt.xlabel('Time',size=25)
plt.ylabel('Amplitude',size=25)
plt.rc('xtick', labelsize=25)   
plt.rc('ytick', labelsize=25)
plt.show()
fig.savefig('LNGConvNetRNNPoint.eps',dpi=600)

#%% NLGKalmanConvNetRNNPoint
i = 0
convNet = np.load(r'C:\Project16\resultData\NLGConvNetSmoothed.dat')[i]
rnn = np.load(r'C:\Project16\resultData\NLGRNNSmoothed.dat')[i]
zTest = np.load(r'C:\Project16\generateData\NLG200zTest.dat')[i,:,0]

fig = plt.figure(figsize=(10,7))
plt.plot(convNet,color='green',ls='-',label='ConvNet Point Estimator',lw=3)
plt.plot(rnn,color='red',ls='--',dashes=(3,3),label='RNN Point Estimator',lw=3)
plt.scatter(np.arange(kalman.shape[0]),zTest,label='True Data',color='black',s=7)
plt.subplots_adjust(left=0.11,bottom=0.12,right=0.98,top=0.95)
plt.legend(prop={'size': 20})
plt.xlabel('Time',size=25)
plt.ylabel('Amplitude',size=25)
plt.rc('xtick', labelsize=25)   
plt.rc('ytick', labelsize=25)
plt.show()
fig.savefig('NLGConvNetRNNPoint.eps',dpi=600)

#%% NLNGKalmanConvNetRNNPoint
i = 0
convNet = np.load(r'C:\Project16\resultData\NLNGConvNetSmoothed.dat')[i]
rnn = np.load(r'C:\Project16\resultData\NLNGRNNSmoothed.dat')[i]
zTest = np.load(r'C:\Project16\generateData\NLNG200zTest.dat')[i,:,0]

fig = plt.figure(figsize=(10,7))
plt.plot(convNet,color='green',ls='-',label='ConvNet Point Estimator',lw=2)
plt.plot(rnn,color='red',ls='--',dashes=(3,3),label='RNN Point Estimator',lw=2)
plt.scatter(np.arange(kalman.shape[0]),zTest,label='True Data',color='black',s=7)
plt.subplots_adjust(left=0.11,bottom=0.12,right=0.98,top=0.95)
plt.legend(prop={'size': 20})
plt.xlabel('Time',size=25)
plt.ylabel('Amplitude',size=25)
plt.rc('xtick', labelsize=25)   
plt.rc('ytick', labelsize=25)
plt.show()
fig.savefig('NLNGConvNetRNNPoint.eps',dpi=600)

#%% BBVI mean, train then test
i = 0
meanRNN = np.load(r'C:\Project16\resultData\BBVI_output_mu_testData.dat')[:,i]
zTest = np.load(r'C:\Project16\generateData\LG200zTest.dat')[i,:,0]
meanKalman = np.load(r'C:\Project16\resultData\LGmeanKalman.dat')[i]


fig = plt.figure(figsize=(10,7))
plt.plot(meanKalman,color='blue',linewidth='3',label='Kalman')
plt.plot(meanRNN,color='red',ls='--',dashes=(3,3),lw=3,label='BBVI')
plt.scatter(np.arange(200),zTest,label='True Data',color='black')
plt.subplots_adjust(left=0.11,bottom=0.12,right=0.98,top=0.95)
plt.legend(prop={'size': 20})
plt.xlabel('Time',size=25)
plt.ylabel('Expected Value',size=25)
plt.rc('xtick', labelsize=25)    
plt.rc('ytick', labelsize=25)
plt.show()
fig.savefig('rnnMeanTrainThenTest.eps',dpi=600)

#%% BBVI std, train then test
i = 0
stdRNN = np.load(r'C:\Project16\resultData\BBVI_output_sigma_testData.dat')[:,i]
stdKalman = np.sqrt(np.load(r'C:\Project16\resultData\LGCovKalman.dat')[i])
zTest = np.load(r'C:\Project16\generateData\LG200zTest.dat')
zTest = zTest[0]
fig = plt.figure(figsize=(10,7))
plt.plot(stdKalman,color='blue',linewidth='3',label='Kalman')
plt.plot(stdRNN,color='red',ls='--',dashes=(3,3),lw=3,label='BBVI')
plt.subplots_adjust(left=0.11,bottom=0.12,right=0.98,top=0.95)
plt.legend(prop={'size': 20})
plt.xlabel('Time',size=25)
plt.ylabel('Standard Deviation',size=25)
plt.rc('xtick', labelsize=25)    
plt.rc('ytick', labelsize=25)
plt.show()
fig.savefig('rnnStdTrainThenTest.eps',dpi=600)

#%% rnnELBO
elbo = np.load(r'C:\Project16\resultData\rnnELBO.dat')
fig = plt.figure(figsize=(10,7))
plt.plot(elbo,color='blue',lw=2)
plt.subplots_adjust(left=0.1,bottom=0.1,right=0.98,top=0.95)
plt.legend(prop={'size': 25})
plt.xlabel('Iteration',size=25)
plt.ylabel('ELBO',size=25)
plt.show()

fig.savefig('rnnELBO.eps',dpi=600)

