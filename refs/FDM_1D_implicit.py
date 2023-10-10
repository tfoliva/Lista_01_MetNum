import numpy as np
import matplotlib.pyplot as plt

## Input data
# Geometry and material pproperties
L = 0.1
rho = 8100
c = 500
kcond = 17
# Boudary conditions
Tbar = lambda t: 100 + 5 * t / 60
hc = 10
Tinf = 50
# Initial conditions
T0 = 100
# Final time
tend = 10

## Discretization
N = 20  # spatial
Nt = 100  # temporal
x = np.linspace(0, L, N)
t = np.linspace(0, tend, Nt)
dx = x[1] - x[0]
dt = t[1] - t[0]

beta = kcond * dt / (rho * c * dx**2)

T = np.zeros((N, Nt))

# Matrix A assembly
A = np.zeros((N, N))

for i, xi in enumerate(x):
    if np.isclose(xi, 0):
        A[0, 0] = 1
    elif np.isclose(xi, L):
        A[i, i - 1] = -2 * kcond / dx**2
        A[i, i] = rho * c / dt + 2 * kcond / dx**2 + hc / dx
    else:
        A[i, i - 1] = -kcond / dx**2
        A[i, i + 1] = -kcond / dx**2
        A[i, i] = rho * c / dt + 2 * kcond / dx**2

Ainv = np.linalg.inv(A)

# Solution
for k, tk in enumerate(t):
    if np.isclose(tk, 0):
        T[:, k] = T0
    else:
        b = (rho * c / dt) * T[:, k - 1]
        b[0] = Tbar(tk)
        b[-1] += (2 * hc / dx) * Tinf
        T[:, k] = Ainv @ b

plt.pcolor(t, x, T[:-1, :-1], cmap="hot")
plt.colorbar()
plt.show()
