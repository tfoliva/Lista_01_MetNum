import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1
from fdcoeff import approx_derivative
import time

start = time.time()

# Questão 03
def f(x):
    return j0(x)


# first derivative of J0 is -J1
def f_1p(x):
    return -j1(x)


# encontre em x = 3
h = 0.25
xbar = 3

# exact solution
exact = f_1p(xbar)
print(f"Exact solution. f'(x) = {exact}")

# (a) backward com 2 pontos:
xb2 = [xbar - h, xbar]
approx = approx_derivative(1, xbar, xb2, j0)[0]
print(f"a: approx {approx}, error (exact - approx) {exact - approx:.2E}\n")

# (b) backward com 3 pontos:
xb3 = [xbar - 2 * h, xbar - h, xbar]
approx = approx_derivative(1, xbar, xb3, j0)[0]
print(f"b: approx {approx}, error (exact - approx) {exact - approx:.2E}\n")

# (c) forward com 2 pontos:
xf2 = [xbar, xbar + h]
approx = approx_derivative(1, xbar, xf2, j0)[0]
print(f"c: approx {approx}, error (exact - approx) {exact - approx:.2E}\n")

# (d) forward com 3 pontos:
xf3 = [xbar, xbar + h, xbar + 2 * h]
approx = approx_derivative(1, xbar, xf3, j0)[0]
print(f"d: approx {approx}, error (exact - approx) {exact - approx:.2E}\n")

# (e) central com 2 pontos:
xc2 = [xbar - h, xbar + h]
approx = approx_derivative(1, xbar, xc2, j0)[0]
print(f"e: approx {approx}, error (exact - approx) {exact - approx:.2E}\n")

# (f) central com 4 pontos:
xc4 = [xbar - 2 * h, xbar - h, xbar + h, xbar + 2 * h]
approx = approx_derivative(1, xbar, xc4, j0)[0]
print(f"f: approx {approx}, error (exact - approx) {exact - approx:.2E}\n")
end = time.time()

print(f"Q3 took {end-start} seconds")
