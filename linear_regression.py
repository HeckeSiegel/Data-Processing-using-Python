import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import math as m

#read in data 
data = np.array([[float(i) for i in (' '.join(line.split())).split(' ')] for line in open('NO2_DOAS_data.txt').readlines()[1:]])
t = np.array((data[:,0]-735090)*24) #transform date variables into hours
dates = dates.num2date(data[:,0])

#plotting
colors =("b","g","r")
labels = ("Langham","School","Street")
plt.xticks(rotation=30)

#plot data with measured errors
def printErrorbar():
    for i in range(3):
        plt.errorbar(dates,data[:,2*i+1],yerr=data[:,2*i+2],color=colors[i],linewidth=2,label=labels[i])

#ordinary least square
def vandermonde(t):
    A = np.vstack((np.ones(t.shape[0]),t)).T
    return A

def regressionParameter(A,b):
    x = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
    return x

def printRegression():
    t = np.array((data[:,0]-735090)*24)
    A = vandermonde(t)
    for i in range(3):
        x = regressionParameter(A,data[:,2*i+1])
        f = x[0] + x[1]*t
        plt.plot(dates,f,'--',color=colors[i],linewidth=0.5,label='y = '+str(x[0])[:5]+str(x[1])[:5]+'*x')

#weighted least square
def weightedLeastSquare(A,y,b):
    W = np.diag(y**(-2))
    x = np.linalg.inv(A.T.dot(W).dot(A)).dot(A.T).dot(W).dot(b)
    return x

def printWeightedRegression():
    t = np.array((data[:,0]-735090)*24)
    A = vandermonde(t)
    for i in range(3):
       x = weightedLeastSquare(A,data[:,2+2*i],data[:,1+2*i])
       f = x[0] + x[1]*t
       plt.plot(dates,f,':',color=colors[i],label='y = '+str(x[0])[:5]+str(x[1])[:5]+'*x')

#total least square
def augmentedMatrix(t,b):
    C = np.vstack((np.ones(t.shape[0]),t,b)).T
    return C

def totalLeastSquare(i):
    t = np.array((data[:,0]-735090)*24)
    C = augmentedMatrix(t,data[:,2*i+1])
    U,W,V = np.linalg.svd(C, full_matrices=True)
    VAb = V.T[0:2,2]
    Vbb = V.T[2,2]
    x = -VAb/Vbb
    return x

def printTotalLeastSquare():
    t = np.array((data[:,0]-735090)*24)
    for i in range(3):
        x = totalLeastSquare(i)
        f = x[0] + x[1]*t
        plt.plot(dates,f,colors[i]+'-',linewidth=0.8,label='y = '+str(x[0])[:5]+str(x[1])[:5]+'*x')
        
#uncertainties for linear regression
def printUncertainty(x,i,n,title):
    e = x[0]+x[1]*t-data[:,2*i+1]
    sigmae = (np.sum(e**2)/(n-2))**0.5
    sigmax2 = sigmae/((np.var(t))**0.5)
    sigmax1 = sigmax2*(np.mean(t**2)/n)**0.5
    f = sigmax1+sigmax2*t
    plt.plot(dates,f,label=title)
    
def printFancy(xlabel,title):
    plt.xlabel('time')
    plt.ylabel(xlabel)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()
   
printErrorbar()
printRegression()
printFancy('NO2 mixing ratio (ppbv)','Ordinary Least Square')

printErrorbar()
printWeightedRegression()
printFancy('NO2 mixing ratio (ppbv)','Weighted Least Square')

printErrorbar()
printTotalLeastSquare()
printFancy('NO2 mixing ratio (ppbv)','Total Least Square')

printRegression()
printWeightedRegression()
printTotalLeastSquare()
printFancy('NO2 mixing ratio (ppbv)','All three methods compared')

t = np.array((data[:,0]-735090)*24)
A = vandermonde(t)
n = len(t)
i = 0
x1 = regressionParameter(A,data[:,2*i+1])
printUncertainty(x1,i,n,'ordinary')
x2 = weightedLeastSquare(A,data[:,2+2*i],data[:,1+2*i])
printUncertainty(x2,i,n,'weighted')
x3 = totalLeastSquare(i)
printUncertainty(x3,i,n,'total')
printFancy('error variance (ppbv)','Error variance for all 3 methods')
