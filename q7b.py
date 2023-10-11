import numpy as np
import matplotlib.pyplot as plt

## Input data
#Geometry and material properties
Lix = -1
Liy = -1
Lfx = 1
Lfy = 1
lamb = 0.01

# Final time
tend = 100

## Discretization
Nx = 25 # spatial x
Ny = 25 # spatial y
Nt = 1001 # temporal
x = np.linspace(Lix, Lfx, Nx)
y = np.linspace(Liy, Lfy, Ny)
t = np.linspace(0, tend, Nt)
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = t[1] - t[0]

beta1 = lamb * dt / (dx**2)
beta2 = lamb * dt / (dy**2)

# Initial conditions
T0 = np.zeros((Nx,Ny))
for i, xi in enumerate(x):
    for j, yj in enumerate(y):
        T0[i,j] = 10 * np.exp((-20) * (xi**2 + yj**2))

# Flatten important matrices 
[xg, yg] = np.meshgrid(x, y)
x = xg.flatten()
y = yg.flatten()
T0 = T0.flatten()

# print(xg)
# print(yg)
# print(x)
# print(y)

# regions
region_B = np.arange(0, Nx, 1)
region_R = np.arange(Nx - 1, Nx * Ny, Nx)
region_T = np.arange(Nx * (Ny - 1), Nx * Ny,1)
region_L = np.arange(0, Nx * (Ny - 1) + 1, Nx)

# Matrix A assembly
A = np.zeros((Nx*Ny,Nx*Ny))

for i, xi in enumerate(x):
    # Vertices
    if i in (set(region_L) & set(region_B)):
        A[i,i] = 2 * beta1 + 2 * beta2 + 1
        A[i,i+1] = -2* beta1
        A[i,i+Nx] = -2 * beta2
    elif i in (set(region_L) & set(region_T)):
        A[i,i-Nx] = -2 * beta2
        A[i,i] = 2 * beta1 + 2 * beta2 + 1
        A[i,i+1] = -2* beta1
    elif i in (set(region_R) & set(region_B)):
        A[i,i-1] = -2 * beta1
        A[i,i] = 2 * beta1 + 2 * beta2 + 1
        A[i,i+Nx] = -2 * beta2
    elif i in (set(region_R) & set(region_T)):
        A[i,i-Nx] = -2 * beta2
        A[i,i-1] = -2 * beta1
        A[i,i] = 2 * beta1 + 2 * beta2 + 1
    # Edges without vertices
    elif i in set(region_L) - (set(region_L) & set(region_B)) - (set(region_L) & set(region_T)):
        A[i,i-Nx] = -beta2
        A[i,i] = 2 * beta1 + 2 * beta2 + 1
        A[i,i+1] = -2* beta1
        A[i,i+Nx] = -beta2
    elif i in set(region_R) - (set(region_R) & set(region_B)) - (set(region_R) & set(region_T)):
        A[i,i-Nx] = -beta2
        A[i,i-1] = -2 * beta1
        A[i,i] = 2 * beta1 + 2 * beta2 + 1
        A[i,i+Nx] = -beta2
    elif i in set(region_B) - (set(region_B) & set(region_L)) - (set(region_B) & set(region_R)):
        A[i,i-1] = -beta1
        A[i,i] = 2 * beta1 + 2 * beta2 + 1
        A[i,i+1] = -beta1
        A[i,i+Nx] = -2 * beta2
    elif i in set(region_T) - (set(region_T) & set(region_L)) - (set(region_T) & set(region_R)):
        A[i,i-Nx] = -2 * beta2
        A[i,i-1] = -beta1
        A[i,i] = 2 * beta1 + 2 * beta2 + 1
        A[i,i+1] = -beta1
    # Other positions
    else:
        A[i,i-Nx] = -beta2
        A[i,i-1] = -beta1
        A[i,i] = 2 * beta1 + 2 * beta2 + 1
        A[i,i+1] = -beta1
        A[i,i+Nx] = -beta2

Ainv = np.linalg.inv(A)

# print(A)

T = np.zeros((Nx*Ny,Nt))
b = np.zeros(Nx*Ny)
# Solution
for k, tk in enumerate(t):
    if tk == 0:
        T[:,k] = T0
    else:
        b = list(T[:,k-1])
        T[:,k] = Ainv @ b


Tmatrix = np.zeros((Nt,Ny,Nx))
for k in range(Nt):
    Tmatrix[k,:,:] = np.reshape(T[:,k],(Ny,Nx))

plt.figure()
plt.contourf(xg, yg, Tmatrix[0,:,:], 20, cmap='hot')
plt.colorbar()
plt.title('t = 0')

plt.figure()
plt.contourf(xg, yg, Tmatrix[250,:,:], 20, cmap='hot')
plt.colorbar()
plt.title('t = 25')

plt.figure()
plt.contourf(xg, yg, Tmatrix[500,:,:], 20, cmap='hot')
plt.colorbar()
plt.title('t = 50')

plt.figure()
plt.contourf(xg, yg, Tmatrix[750,:,:], 20, cmap='hot')
plt.colorbar()
plt.title('t = 75')

plt.figure()
plt.contourf(xg, yg, Tmatrix[1000,:,:], 20, cmap='hot')
plt.colorbar()
plt.title('t = 100')

# Plotting the solution
xplot = np.linspace(Lix, Lfx, Nx)
yplot = np.linspace(Liy, Lfy, Ny)
plt.figure()
for k,tk in enumerate(t):
    # if tk == 0:
    #     plt.plot(xplot,Tmatrix[k,:,((Ny-1)//2)],'r')
    if tk == 25:
        plt.plot(xplot,Tmatrix[k,:,((Ny-1)//2)],'darkorange')
    elif tk == 50:
        plt.plot(xplot,Tmatrix[k,:,((Ny-1)//2)],'y')
    elif tk == 75:
        plt.plot(xplot,Tmatrix[k,:,((Ny-1)//2)],'g')
    elif tk == 100:
        plt.plot(xplot,Tmatrix[k,:,((Ny-1)//2)],'blue')
    else:
        continue
plt.title('Temperature profile (y=0)')
plt.xlabel('x')
plt.ylabel('Temperature')
plt.show()
