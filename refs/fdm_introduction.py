# -*- coding: utf-8 -*-
"""FDM_introduction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dZb-11EHgnf6kzXakBt1HZCKEIXL5i6X

FDM - 1D - Dirichlet boundary conditions in both sides
"""

import numpy as np
import matplotlib.pyplot as plt

# Input data
L = 1
s = 20
k = 0.1
T1bar = 100
T2bar = 50

# Discretization
N = 4
x = np.linspace(0, L, N)
h = x[1] - x[0]

# Assembly of matrices and vector
A = np.diag(np.ones(N) * 2)
A = A + np.diag(np.ones(N-1) * (-1), 1)
A = A + np.diag(np.ones(N-1) * (-1), -1)
b = (s * h**2.0/k) * np.ones(N)

# Boundary conditions
A[0, :] = 0
A[0, 0] = 1
b[0] = T1bar

A[N-1, :] = 0
A[N-1, N-1] = 1
b[N-1] = T2bar

# Solving
T = np.linalg.solve(A, b)

# Reference solution
xref = np.linspace(0, L, 200)
Tref = -s*xref**2/(2*k) + (T2bar - T1bar) * xref / L + s * L * xref / (2 * k) + T1bar

# Plotting the solution
plt.plot(xref, Tref, '-k', label='Reference')
plt.plot(x, T, '--sb', label='FDM')
plt.xlabel('x')
plt.ylabel('Temperature [K]')
plt.legend()
plt.show()

"""FDM - 1D - Convection in x = L"""

import numpy as np
import matplotlib.pyplot as plt

# Input data
L = 1
s = 20
k = 0.1
T1bar = 100
hc = 10
Tinf = 20

# Discretization
N = 4
x = np.linspace(0, L, N)
h = x[1] - x[0]

# Assembly of matrices and vector
A = np.diag(np.ones(N) * 2)
A = A + np.diag(np.ones(N-1) * (-1), 1)
A = A + np.diag(np.ones(N-1) * (-1), -1)
b = (s * h**2.0/k) * np.ones(N)

# Boundary conditions
# T(x = 0) = T1bar
A[0, :] = 0
A[0, 0] = 1
b[0] = T1bar

# -k * dTdx(x = L) = hc * (T(x=L) - Tinf)
A[N-1, N-2] += -1
A[N-1, N-1] += 2 * hc * h / k
b[N-1] += 2 * hc * h * Tinf / k

# Solving
T = np.linalg.solve(A, b)

# Plotting the solution
plt.plot(x, T, '--sb', label='FDM')
plt.xlabel('x')
plt.ylabel('Temperature [K]')
plt.legend()
plt.show()

"""FDM - 1D - Gaussiana"""

import numpy as np
import matplotlib.pyplot as plt

# Input data
L = 10e-3
s0 = 1e6
sigma = 1e-7
k = 0.2
hc = 30
Tinf = 25

# Discretization
N = 200
x = np.linspace(0, L, N)
h = x[1] - x[0]

# Assembly of matrices and vector
A = np.diag(np.ones(N) * 2)
A = A + np.diag(np.ones(N-1) * (-1), 1)
A = A + np.diag(np.ones(N-1) * (-1), -1)
s = lambda x: s0 * np.exp(-x**2 / sigma)
b = s(x) * h**2.0/k

# Boundary conditions
# q(x = 0) = 0 - Adiabatic
A[0, 1] += -1

# -k * dTdx(x = L) = hc * (T(x=L) - Tinf)
A[N-1, N-2] += -1
A[N-1, N-1] += 2 * hc * h / k
b[N-1] += 2 * hc * h * Tinf / k

# Solving
T = np.linalg.solve(A, b)

# Plotting the solution
plt.plot(x, T, '--sb', label='FDM')
plt.xlabel('x')
plt.ylabel('Temperature [K]')
plt.legend()
plt.show()

"""FDM - 2D"""

import numpy as np
import matplotlib.pyplot as plt

# Input data
Lx = 2
Ly = 1
k = 0.3
s = 20
Tbar = 50

# Discretization
Nx = 50
Ny = 25
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
hx = x[1] - x[0]
hy = y[1] - y[0]
[xg, yg] = np.meshgrid(x, y)
x = xg.flatten()
y = yg.flatten()

# regions
region_B = np.arange(0, Nx, 1)
region_R = np.arange(Nx - 1, Nx * Ny, Nx)
region_T = np.arange(Nx * (Ny - 1), Nx * Ny)
region_L = np.arange(0, Nx * (Ny - 1) + 1, Nx)

'''
plt.plot(x[region_B], y[region_B], 'sb', label='Bottom')
plt.plot(x[region_R], y[region_R], 'sr', label='Right')
plt.plot(x[region_T], y[region_T], 'sg', label='Top')
plt.plot(x[region_L], y[region_L], 'sm', label='Left')
plt.plot(x, y, '.k')
plt.legend()
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.text(xi, yi, ' ' + str(i))
plt.show()
'''

# Assembly of matrices and vectors
A = np.zeros((Nx * Ny, Nx * Ny))
b = np.zeros(Nx * Ny)

for i in range(Nx * Ny):
    if i in region_B:
        A[i, i] = 1
        b[i] = Tbar
    elif i in region_R:
        A[i, i] = 1
        b[i] = Tbar
    elif i in region_T:
        A[i, i] = 1
        b[i] = Tbar
    elif i in region_L:
        A[i, i] = 1
        b[i] = Tbar
    else:
        A[i, i - Nx] = -1 / hy**2
        A[i, i - 1] = -1 / hx**2
        A[i, i] = 2 / hx**2 + 2 / hy**2
        A[i, i + 1] = -1 / hx**2
        A[i, i + Nx] = -1 / hy**2
        b[i] = s / k

# Solving
T = np.linalg.solve(A, b)
Tmatrix = T.reshape((Ny, Nx))

# Plotting
plt.figure()
plt.pcolor(Tmatrix)
plt.colorbar()

# Matrix sparsity
f = np.sum(np.abs(A) > np.finfo(float).eps) / (Nx * Ny)**2
print(f)