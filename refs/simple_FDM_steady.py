#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Input data
L = 1 # thickness
T1 = 20 # T(x=0)
T2 = 10 # T(x=L)
s = 2
k = 0.1 # Thermal conductivity

# Discretization
n = 20

x = np.linspace(0, L, n)
h = x[1] - x[0]

# Assembly of the global matrix and vector
diag1 = -2 * np.ones(n)
diag2 = np.ones(n-1)

A = np.diag(diag1, 0)
A = A + np.diag(diag2, 1)
A = A + np.diag(diag2,-1)

f = -(s/k) * h**2.0 * np.ones(n)

# Boundary conditions
A[0,:] = 0.0
A[0,0] = 1.0

A[-1,:] = 0.0
A[-1,-1] = 1.0

f[0] = T1
f[-1] = T2

# Solution of the linear system
T = np.linalg.solve(A, f)

# Plot the solution
plt.close('all')
plt.plot(x,T, '-ob')
plt.xlabel('x')
plt.ylabel('T')