import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import math as m
import scipy.optimize

#read in, tidy and plot data
data0 = np.array([[float(i) for i in (' '.join(line.split())).split(' ')] for line in open('higher_order_linear_regression.txt').readlines()[1:]])
data = []
for i in range(0,len(data0),1):
    if (np.isnan(data0[i,1]) == False and np.isnan(data0[i,2]) == False):
        data.append(data0[i])   
data = np.array(data)       
plt.subplot(211)
plt.errorbar(data[:,0],data[:,1],yerr=data[:,2],color='green',linewidth=0.5,label='NO2-measurement')
plt.plot(data[:,0],data[:,3],color='red',linewidth=0.5,label='2-year average NO2')
plt.fill_between(data[:,0],data[:,3]+data[:,4],data[:,3]-data[:,4],color='grey',alpha=0.3,label='error')

#regression
b = data[:,1]
t = data[:,3]

##first order
A1 = np.vstack((np.ones(t.shape[0]),t)).T
x1 = np.linalg.inv(A1.T.dot(A1)).dot(A1.T).dot(b)
f1 = x1[0] + x1[1]*data[:,3]
plt.plot(data[:,0],f1,color='blue',label='1st order fit')
epsilon1 = np.abs(b-A1.dot(x1))

##second order
A2 = np.vstack((np.ones(t.shape[0]),t,t*data[:,0])).T
x2 = np.linalg.inv(A2.T.dot(A2)).dot(A2.T).dot(b)
f2 = x2[0] + t*(x2[1]+x2[2]*data[:,0])
plt.plot(data[:,0],f2,color='black',label='2nd order fit')
epsilon2 = np.abs(b-A2.dot(x2))

plt.xlabel('time')
plt.ylabel('NO2 mixing ratio (ppb)')
plt.grid()
plt.legend()
plt.subplot(212)
plt.plot(data[:,0],epsilon1,color='blue',label='1st order error')
plt.plot(data[:,0],epsilon2,color='black',label='2nd order error')
plt.xlabel('time')
plt.ylabel('error term (ppb)')
plt.grid()
plt.legend()

plt.show()
