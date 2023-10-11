import numpy as np
import matplotlib.pyplot as plt

## Input data
#Geometry and material properties
L = 1
lamb = 0.01
#Boudary conditions
q = 1
# Initial conditions
T0 = 1
# Final time
tend = 30

## Discretization
N = 50 # spatial
Nt = 3000 # temporal
x = np.linspace(0, L, N)
t = np.linspace(0, tend, Nt)
dx = x[1] - x[0]
dt = t[1] - t[0]

beta = lamb * dt / (dx**2)

T = np.zeros((N,Nt))
for k, tk in enumerate(t):
    for i, xi in enumerate(x):
        if tk == 0:
            T[i,k] = T0
        else:
            if np.isclose(xi,0):
                if t[k] <= 10: 
                    T[i,k] = T[i,k-1] + beta * (2 * T[i+1,k-1] - 2 * T[i,k-1] + 2 * dx * q)
                else:
                     T[i,k] = T[i,k-1] + beta * (2 * T[i+1,k-1] - 2 * T[i,k-1])
            elif np.isclose(xi,L):
                T[i,k] = T[i,k-1] + beta * (2 * T[i-1,k-1] - 2 * T[i,k-1])
            else:
                T[i,k] = T[i,k-1] + beta * (T[i-1,k-1] - 2 * T[i,k-1] + T[i+1,k-1])


print(T)

plt.figure()
plt.pcolor(x,t,T.T,cmap='hot')
plt.colorbar()
plt.savefig("figs/q6a_colormap.png", dpi=300)


# Plotting the solution
plt.figure()
for k,tk in enumerate(t):
    if k % 100 == 0:
        if tk <= 10:
            plt.plot(x,T[:,k],'r')
        else:
            plt.plot(x,T[:,k],'b')
    else:
        continue
plt.title('Temperature profile')
plt.xlabel('x')
plt.ylabel('Temperature')
plt.savefig("figs/q6a_temperature_profile.png", dpi=300)
