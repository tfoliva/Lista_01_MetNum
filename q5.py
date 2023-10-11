import numpy as np
import matplotlib.pyplot as plt

## Input data
#Geometry and material properties
Lix = -0.5 # m
Liy = -0.5 # m
Lfx = 0.5 # m
Lfy = 0.5 # m
k = 0.5 # W/(mK)


# Boundary conditions
q_bar = 500 # W/m^2
h = 10 # W/(m^2 K)
T_inf = 0 # K

# Local error
tau = 10 ** -4
h_error = np.sqrt(tau) #Constant aprox. 1

## Discretization
Nx = round((Lfx-Lix)/h_error + 1) # spatial x
Ny = round((Lfy-Liy)/h_error + 1) # spatial y
x = np.linspace(Lix, Lfx, Nx)
y = np.linspace(Liy, Lfy, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

print(dx)

# Intermediate constants
beta1 = k / (dx**2)
beta2 = k / (dy**2)

gamma1 = 2 * dx * q_bar / k
gamma2 = 2 * dx * h / k
gamma3 = 2 * dy * h / k

# Flatten important matrices 
[xg, yg] = np.meshgrid(x, y)
x = xg.flatten()
y = yg.flatten()

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
        A[i,i] = -(2 * beta1 + (2 + gamma3) * beta2)
        A[i,i+1] = 2 * beta1
        A[i,i+Nx] = 2 * beta2
    elif i in (set(region_L) & set(region_T)):
        A[i,i-Nx] = 2 * beta2
        A[i,i] = -(2 * beta1 + (gamma3 + 2) * beta2)
        A[i,i+1] = 2 * beta1
    elif i in (set(region_R) & set(region_B)):
        A[i,i-1] = 2 * beta1
        A[i,i] = -((gamma2 + 2) * beta1 + (2 + gamma3) * beta2)
        A[i,i+Nx] = 2 * beta2
    elif i in (set(region_R) & set(region_T)):
        A[i,i-Nx] = 2 * beta2
        A[i,i-1] = 2 * beta1
        A[i,i] = -((gamma2 + 2) * beta1 + (gamma3 + 2) * beta2)
    # Edges without vertices
    elif i in set(region_L) - (set(region_L) & set(region_B)) - (set(region_L) & set(region_T)):
        A[i,i-Nx] = beta2
        A[i,i] = -2 * (beta1 + beta2)
        A[i,i+1] = 2 * beta1
        A[i,i+Nx] = beta2
    elif i in set(region_R) - (set(region_R) & set(region_B)) - (set(region_R) & set(region_T)):
        A[i,i-Nx] = beta2
        A[i,i-1] = 2 * beta1
        A[i,i] = -((gamma2 + 2) * beta1 + 2 * beta2)
        A[i,i+Nx] = beta2
    elif i in set(region_B) - (set(region_B) & set(region_L)) - (set(region_B) & set(region_R)):
        A[i,i-1] = beta1
        A[i,i] = -(2 * beta1 + (2 + gamma3) * beta2)
        A[i,i+1] = beta1
        A[i,i+Nx] = 2 * beta2
    elif i in set(region_T) - (set(region_T) & set(region_L)) - (set(region_T) & set(region_R)):
        A[i,i-Nx] = 2 * beta2
        A[i,i-1] = beta1
        A[i,i] = -(2 * beta1 + (gamma3 + 2) * beta2)
        A[i,i+1] = beta1
    # Other positions
    else:
        A[i,i-Nx] = beta2
        A[i,i-1] = beta1
        A[i,i] = -2 * (beta1 + beta2)
        A[i,i+1] = beta1
        A[i,i+Nx] = beta2

Ainv = np.linalg.inv(A)

# print(A)

# Vector b assembly

b = np.zeros(Nx*Ny)

for i, xi in enumerate(x):
    # Vertices
    if i in (set(region_L) & set(region_B)):
        b[i] = -beta1 * gamma1 - beta2 * gamma3 * T_inf
    elif i in (set(region_L) & set(region_T)):
        b[i] = -beta1 * gamma1 - beta2 * gamma3 * T_inf
    elif i in (set(region_R) & set(region_B)):
        b[i] = -beta1 * gamma2 * T_inf - beta2 * gamma3 * T_inf
    elif i in (set(region_R) & set(region_T)):
        b[i] = -beta1 * gamma2 * T_inf - beta2 * gamma3 * T_inf
    # Edges without vertices
    elif i in set(region_L) - (set(region_L) & set(region_B)) - (set(region_L) & set(region_T)):
        b[i] = -beta1 * gamma1
    elif i in set(region_R) - (set(region_R) & set(region_B)) - (set(region_R) & set(region_T)):
        b[i] = -beta1 * gamma2 * T_inf 
    elif i in set(region_B) - (set(region_B) & set(region_L)) - (set(region_B) & set(region_R)):
        b[i] = -beta2 * gamma3 * T_inf
    elif i in set(region_T) - (set(region_T) & set(region_L)) - (set(region_T) & set(region_R)):
        b[i] = -beta2 * gamma3 * T_inf
    # Other positions
    else:
        b[i] = 0

# print(b)

# Solve the system
T = np.zeros((Nx*Ny,1))
T = Ainv @ b

# print(T)

Tmatrix = np.reshape(T,(Ny,Nx))

# print(Tmatrix)

plt.figure()
plt.contourf(xg, yg, Tmatrix, 30, cmap='hot')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.savefig("figs/q5_colorbar.png", dpi=300)

# Heat flux on the boundaries

q = np.zeros(4)
for i, xi in enumerate(x):
    if i in set(region_L):
        q[0] = q_bar
    elif i in set(region_R):
        dq = -h * dy * (T[i] - T_inf)
        q[1] = q[1] + dq
    elif i in set(region_B):
        dq = -h * dx * (T[i] - T_inf)
        q[2] = q[2] + dq
    elif i in set(region_T):
        dq = -h *  dx * (T[i] - T_inf)
        q[3] = q[3] + dq
    else:
        continue

print(q)
print(q[0] + q[1] + q[2] + q[3])


