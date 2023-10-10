#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
#                               Input data
# ------------------------------------------------------------------------------
# Geometry and material pproperties
L = 1
rho = 8100
c = 500
kcond = 17
alpha = kcond / (rho * c)
# Boudary conditions
Tbar = lambda t: 100 + 5 * t / 60
hc = 10
Tinf = 50
# Initial conditions
T0 = 100
# Final time
tend = 30

# Discretization
N = 20  # spatial
Nt = 100  # temporal

x = np.linspace(0, L, N)
t = np.linspace(0, tend, Nt)
dx = x[1] - x[0]
dt = t[1] - t[0]


# ------------------------------------------------------------------------------
#                               Function definitions
# ------------------------------------------------------------------------------
def q(t):
    if t <= 10:
        return 1.0
    else:
        return 0.0


def dT_dt(temperature_k_plus_1, temperature_k, timestep):
    return (temperature_k_plus_1 - temperature_k) / timestep


# ------------------------------------------------------------------------------
#                       Assembly of the global matrix and vector
# ------------------------------------------------------------------------------
off_diagonal_coeff = -alpha * dt / dx
diagonal_coeff = 1 - 2 * off_diagonal_coeff

T = np.zeros((N, Nt))

# Matrix A assembly
A = np.zeros((N, N))

for i, xi in enumerate(x):
    if np.isclose(xi, 0):
        A[0, 0] = diagonal_coeff
    elif np.isclose(xi, L):
        A[i, i - 1] = off_diagonal_coeff
        A[i, i] = 1 + off_diagonal_coeff
    else:
        A[i, i - 1] = off_diagonal_coeff
        A[i, i + 1] = off_diagonal_coeff
        A[i, i] = diagonal_coeff
print(A)

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

plt.pcolor(t, x, T, cmap="hot")
plt.colorbar()
plt.show()
