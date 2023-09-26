#!/usr/bin/env python
# coding: utf-8

# In[2]:


from qutip import *
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from scipy.special import genlaguerre as L
from scipy.integrate import solve_ivp as solve


# To save time and space anytime rho appears without a specific mention of an element of rho (like rhogg), rho will be defined as rho = [rhogg, rhoee, rhoeg]. As such, when we have multiple harmonic osc states, rho[n] = [rhogg, rhoee, rhoeg] and rho = [rho[0], rho[1] ...]
# 
# Update: we need rho to be 1D to be compatible with the integrater. So now rhoggn = rho[3n], rhoeen = rho[3n+1], and rhogen = rho[3n+2]

# In[3]:


def prob(n,nbar):
    return 1.0/(nbar+1)*(nbar/(nbar+1))**n    #returns prob of being in state n given nbar

def dopAndStark(n):       #for 1D!! to make 3d, multiply this by 3                        
    return -2.0*np.pi*nu0*hbar*omega*(n+0.5)/(m*c**2)*(1.0 + alpha*m**2*Omegarf**2*c**2/(h*nu0*e**2*2)) 

def Omega(n,m,eta):
    return np.exp(-eta**2/2.0) * eta**(np.abs(m)) * (math.factorial(n)/math.factorial(n+m))**(np.sign(m)/2.0) \
        * L(n,np.abs(m))(eta**2)              #returns unitless rabi rate of transition n -> n+m given Lamb-Dicke Parameter


# In[33]:


#def groundRho(Ncut):
#    rho = np.array([[0.0+0.0j]*3]*(Ncut+1))
#    rho[0][0] = 1.0+0.0j
#    rho = np.reshape(rho, (Ncut+1)*3)
#    return rho

def groundRho(Ncut, nbar):
    rho = np.array([0.0+0.0j]*3*(Ncut+1))
    for n in range(Ncut):
        rho[3*n] = prob(n,nbar)
    return rho

def heatSys(t, rho, nbardot):
    Ncut = rho.shape[0]//3 - 1
    rhoDot = [0.0+0.0j]*((Ncut+1)*3)
    for ii in range(3):
        rhoDot[0+ii] = nbardot*(-rho[0+ii]+rho[3*1+ii])
        rhoDot[3*Ncut+ii] = nbardot*(-Ncut*rho[3*Ncut+ii]+Ncut*rho[3*(Ncut-1)+ii])
        #Note*** the above line is true in the limit that rho[Ncut+1,ii]=rho[Ncut,ii] (fair assumption for large Ncut I think)
        for n in range(1,Ncut-1):
            rhoDot[3*n+ii] = nbardot*(-(2.0*n+1.0)*rho[3*n+ii] + (n+1.0)*rho[3*(n+1)+ii] + n*rho[3*(n-1)+ii])
    return rhoDot

def freeEvo(t, rho0, delta):
    rhoeg0 = rho0[2]
    rho = [rho0[0], rho0[1], rhoeg0*np.exp(-1.0j*delta*t)]
    return rho

def nbar(rho):
    Ncut = rho.shape[0]//3
    nbar = 0.0
    for n in range(Ncut):
        nbar += n*(rho[n*3].real + rho[n*3+1].real)
    return nbar

def rhogg(rho):
    rhogg = 0.0
    Ncut = rho.shape[0]//3
    for n in range(Ncut):
        rhogg += rho[3*n].real
    return rhogg

def rhoee(rho):
    rhoee = 0.0
    Ncut = rho.shape[0]//3
    for n in range(Ncut):
        rhoee += rho[3*n+1].real
    return rhoee


# In[5]:


def heat(rho, t, nbardot):
    rho = solve(heatSys, [0.0,t], rho, args=[nbardot]).y[:,-1]
    return rho

def ramsey(rho0, T, deld, delPrime, Omega0):
    #T = dark-time, deld = dark-time detuning, delPrime=pulse detuning
    rho = solve(bloch, [0.0, np.pi/(2.0*Omega0)], rho0, args=(Omega0, delPrime)).y[:,-1]
    rho = freeEvo(T, rho, deld)
    rho = solve(bloch, [0.0, np.pi/(2.0*Omega0)], rho, args=(Omega0, delPrime)).y[:,-1]
    return rho


# In[17]:


#takes rho as a 3 element vector and converts it to a 2x2 matrix by calculating the conj. of rhoge
def subpulse(rho0, t, Omega0, delta):
    rho = np.array([[rho0[0], rho0[2]],[np.conj(rho0[2]), rho0[1]]])
    Omega = np.sqrt(Omega0**2 + delta**2)
    if Omega == 0.0:
        U = np.array([[1.0 , 0.0],
                      [0.0 , 1.0]])
    else:
        U = np.array([[np.cos(Omega*t/2.0) -(1.0j*delta/Omega)*np.sin(Omega*t/2.0), (1.0j*Omega0/Omega)*np.sin(Omega*t/2.0)],
                     [(1.0j*Omega0/Omega)*np.sin(Omega*t/2.0) , np.cos(Omega*t/2.0) + (1.0j*delta/Omega)*np.sin(Omega*t/2.0)]])
    rho = U@rho@np.conj(U)
    return np.array([rho[0,0], rho[1,1], rho[0,1]])

def pulse(rho, t, Omega0, delta, eta):
    Ncut = rho.size//3 - 1
    for n in range(Ncut):
        rho[3*n:3*(n+1)] = subpulse(rho[3*n:3*(n+1)] , t, Omega0*Omega(n,0,eta), delta)
    return rho


# In[18]:


groundRho(2)


# In[19]:


pulse(groundRho(2), 1.0, np.pi/2.0, 0.0, 0.0)


# In[101]:


pts = 100
deltas = np.linspace(-25.0, 25.0, pts)
rhoees = np.zeros(pts)
rhoggs = np.zeros(pts)
shift = 5.0
T = 1.0
deld = 0.0

for ii in range(pts):
    rhoees[ii] = ramsey(groundRho(), T, deld, deltas[ii]-shift, 1.0)[1]
    rhoggs[ii] = ramsey(groundRho(), T, deld, deltas[ii]-shift, 1.0)[0]

    
plt.plot(deltas, rhoees)
plt.plot(deltas, rhoees+rhoggs)


# In[103]:


plt.plot(deltas,rhoees-rhoggs)


# In[104]:


sol = solve(bloch, [0,10], [1.0+0.0j,0.0+0.0j,0.0+0.0j], args=(np.pi, 0.0))#, dense_output=True)


# In[54]:


t = np.linspace(0,10,100)
z = sol.sol(t)
plt.plot(t, z.T)
plt.legend(['rhogg','rhoee','rhoge'])


# In[9]:


a = np.array([[1,2],[3,4]])


# In[11]:


a[:,0]


# In[13]:


np.array([1,2,3])


# In[30]:


np.sum(groundRho(100,1.0))


# In[32]:


nbar(groundRho(100,2.0))


# In[40]:


rhogg(groundRho(10000,100.0))


# In[45]:


rhoee(heat(groundRho(1000,1.0),1.0,1.0))


# In[50]:


#def pulse(rho, t, Omega0, delta, eta):
rhoee(pulse(groundRho(1000,1.0), 2.0, np.pi/2.0, 0.0, 0.0))


# In[ ]:




