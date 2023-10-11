import numpy as np
import matplotlib.pyplot as plt

## Input data
# Geometry and material properties
L = 1
lamb = 0.01
# Boudary conditions
q = 1
# Initial conditions
T0 = 1
# Final time
tend = 30

## Discretization
N = 50  # spatial
Nt = 3000  # temporal
x = np.linspace(0, L, N)
t = np.linspace(0, tend, Nt)
dx = x[1] - x[0]
dt = t[1] - t[0]

beta = lamb * dt / (dx**2)

T = np.zeros((N, Nt))

# Matrix A assembly
A = np.zeros((N, N))

for i, xi in enumerate(x):
    if np.isclose(xi, 0):
        A[i, i] = 2 * beta + 1
        A[i, i + 1] = -2 * beta
    elif np.isclose(xi, L):
        A[i, i - 1] = -2 * beta
        A[i, i] = 2 * beta + 1
    else:
        A[i, i - 1] = -beta
        A[i, i + 1] = -beta
        A[i, i] = 2 * beta + 1

Ainv = np.linalg.inv(A)

# print(A)

b = np.zeros(N)
# Solution
for k, tk in enumerate(t):
    if tk == 0:
        T[:, k] = T0
    else:
        b = list(T[:, k - 1])
        if t[k] <= 10:
            b[0] = b[0] + 2 * beta * dx * q
            T[:, k] = Ainv @ b
        else:
            T[:, k] = Ainv @ b


# print(T)


plt.pcolor(x, t, T.T, cmap="hot")
plt.colorbar()
plt.savefig("figs/q6b_colormap.png", dpi=300)

# Plotting the solution
plt.figure()
for k, tk in enumerate(t):
    if k % 100 == 0:
        if tk <= 10:
            plt.plot(x, T[:, k], "r")
        else:
            plt.plot(x, T[:, k], "b")
    else:
        continue
plt.title("Temperature profile")
plt.xlabel("x")
plt.ylabel("Temperature")
plt.savefig("figs/q6b_temperature_profile.png", dpi=300)
