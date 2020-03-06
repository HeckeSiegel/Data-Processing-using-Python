import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def polyvalue(polyparamvec,x):
    y=0
    for i in range(polyparamvec.shape[0]):
       y+=polyparamvec[i]*x**i
    return y

def piecewise_poly(param,n):
    s=param.shape
    vec=np.zeros([s[0]*n])
    for i in range(s[0]):
       vec[i*n:(i+1)*n]=polyvalue(param[i,:],np.linspace(0,1,num=n,endpoint=False))       
    return vec

#random data points to be interpolated
N=6
p=np.random.rand(N+1,1)

#calculate coefficients of polynomials
A=np.zeros([4*N,4*N])
for i in range(N):
    A[i,4*i]=1
    A[N+i,4*i:4*i+4]=[1,1,1,1]
    A[2*N+i,4*i:4*i+4]=[0,1,2,3]
    A[3*N+i,4*i:4*i+4]=[0,0,2,6]
for i in range(N-1):
    A[2*N+i,4*i:4*i+8]=[0,1,2,3,0,-1,0,0]
    A[3*N+i,4*i:4*i+8]=[0,0,2,6,0,0,-2,0]

#boundary conditions
Al = np.zeros(4*N)
Al[2:4] = [2,6]
A[-1,:] = Al

b=np.zeros([4*N,1])
b[:N]=p[:N]
b[N:2*N]=p[1:]
x=np.linalg.inv(A).dot(b)

plt.figure()
Np=100
plt.plot(np.linspace(0,N,num=Np*N),piecewise_poly(np.reshape(x,[N,4]),Np),'r')
plt.plot(p,linestyle='None',marker='x', markersize=10, markeredgewidth=3, color='black')
plt.xlim([-0.1,N+0.1])
plt.show()

