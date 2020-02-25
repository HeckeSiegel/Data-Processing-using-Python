import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np
import math as m

#read in data 
data = np.array([[float(i) for i in (' '.join(line.split())).split(' ')] for line in open('linear_fit_data.txt').readlines()[1:]])
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

#add linear regression lines
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
        plt.plot(dates,f,'--',color=colors[i],linewidth=0.5,label=labels[i]+' lin fit: y = '+str(x[0])[:5]+str(x[1])[:5]+'*x')

#proof that fitlines go through the means of the data
def meanMarkers():
    meanDate = float(str(np.mean(data[:,0]))[:12])
    for i in range(3):
        plt.plot(np.mean(t),np.mean(data[:,2*i+1]),marker='*',color=colors[i],markersize=10)

#uncertainties for linear regression
def printUncertainty():
    t = np.array((data[:,0]-735090)*24)
    A = vandermonde(t)
    n = len(t)
    for i in range(3):
        x = regressionParameter(A,data[:,2*i+1])
        e = x[0]+x[1]*t-data[:,2*i+1]
        sigmae = (np.sum(e**2)/(n-2))**0.5
        sigmax2 = sigmae/((np.var(t))**0.5)
        sigmax1 = sigmax2*(np.mean(t**2)/n)**0.5
        f1 = x[0]+sigmax1*0.1+(x[1]+sigmax2*0.1)*t
        f2 = x[0]-sigmax1*0.1+(x[1]-sigmax2*0.1)*t
        plt.fill_between(dates,f1,f2,color=colors[i],alpha=0.1)

#weighted least square regression
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
       plt.plot(dates,f,':',color=colors[i],label=labels[i]+' err. weighted: y = '+str(x[0])[:5]+str(x[1])[:5]+'*x')

#total least square
def augmentedMatrix(t,b):
    C = np.vstack((np.ones(t.shape[0]),t,b)).T
    return C

def printTotalLeastSquare():
    t = np.array((data[:,0]-735090)*24)
    for i in range(3):
        C = augmentedMatrix(t,data[:,2*i+1])
        U,W,V = np.linalg.svd(C, full_matrices=True)
        VAb = V.T[0:2,2]
        Vbb = V.T[2,2]
        x = -VAb/Vbb
        f = x[0] + x[1]*t
        plt.plot(dates,f,colors[i]+'-',linewidth=0.8,label=labels[i]+'TLS fit y = '+str(x[0])[:5]+str(x[1])[:5]+'*x')
    
def printFancy(title):
    plt.xlabel('time')
    plt.ylabel('NO2 mixing ratio (ppbv)')
    plt.title(title)
    #plt.title('DOAS measurement using 3 different light paths')
    plt.grid()
    plt.legend()
    plt.show()
   
printErrorbar()
printRegression()
printUncertainty()
printFancy('Linear Least Square Regression with Uncertainty')

printErrorbar()
printWeightedRegression()
printFancy('Weighted Least Square Regression')

printErrorbar()
printTotalLeastSquare()
printFancy('Total Least Square Regression')

printRegression()
printWeightedRegression()
printTotalLeastSquare()
printFancy('All three methods compared')
