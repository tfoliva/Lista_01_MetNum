import numpy as np
import math


"""
Find finite difference coefficients

k    - kth derivative order
xbar - target point to approximate around
x    - vector of N stencil points
"""
def fdcoeffV(k, xbar, x):
    if isinstance(x, list):
        x = np.array(x)

    n = len(x)
    A = np.ones((n, n))
    xrow = np.transpose(x - xbar)  # displacements as a row vector.

    for i in range(1, n+1):
        A[i-1, :] = np.divide(np.power(xrow, (i - 1)), math.factorial(i - 1))

    b = np.zeros((n, 1))  # b is right hand side,
    b[k] = 1  # so k’th derivative term remains
    c = np.linalg.solve(A, b)  # solve system for coefficients
    return np.transpose(c)  # row vector

def u_approx(x, c, f, verbose=True):
    if isinstance(x, list):
        x = np.array(x)

    u = f(x)
    u_k = np.dot(c, u)

    if verbose:
        print(f"points: ", x)
        print(f"coeffs: ", c)

        print(f"f(x): ", u)
        print(f"u_k(x): {u_k}\n")
    return u_k

def approx_derivative(k, xbar, x, f):
    coeffs = fdcoeffV(k, xbar, x)
    return u_approx(x, coeffs, f, verbose=False)

if __name__ == "__main__":
    # example for xbar = 3
    # backward with 2 points
    xbar = 3
    h = 1
    x = np.array([xbar, xbar -h, xbar - 2*h])
    k = 1 # first derivative

    print(f"The coefficients are {fdcoeffV(k, xbar, x)}")
