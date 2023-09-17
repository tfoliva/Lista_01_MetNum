import numpy as np
import matplotlib.pyplot as plt

## Input data
#Geometry and material pproperties
L = 0.1
rho = 8100
c = 500
kcond = 17
#Boudary conditions
Tbar = lambda t: 100 + 5 * t / 60
hc = 10 
Tinf = 50
# Initial conditions
T0 = 100
# Final time
tend = 10

## Discretization
N = 20 # spatial
Nt = 100 # temporal
x = np.linspace(0, L, N)
t = np.linspace(0, tend, Nt)
dx = x[1] - x[0]
dt = t[1] - t[0]

beta = kcond * dt / (rho * c * dx**2)

T = np.zeros((N,Nt))
for k, tk in enumerate(t):
    for i, xi in enumerate(x):
        if tk == 0:
            T[i,k] = T0
        else:
            if np.isclose(xi,0):
                T[i,k] = Tbar(tk)
            elif np.isclose(xi,L):
                Tstar = T[i-1,k-1] - 2 * hc * dx * (T[i,k-1] - Tinf) / kcond
                T[i,k] = T[i,k-1] - beta * (-T[i-1,k-1] + 2 * T[i,k-1] - Tstar)
            else:
                T[i,k] = T[i,k-1] - beta * (-T[i-1,k-1] + 2 * T[i,k-1] - T[i+1,k-1])


print(T)

