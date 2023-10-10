import numpy as np
import matplotlib.pyplot as plt
from fdcoeff import fdcoeffV


# Questão 04
# considere f(x) = exp(x) sin(x)
def f(x):
    return np.exp(x)*np.sin(x)

def f_1p(x):
    return np.exp(x)*np.sin(x) + np.exp(x)*np.cos(x)

def f_2p(x):
    return 2*np.exp(x)*np.cos(x)

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

# encontre em x = 2
h = 1
xbar = 2

# (i) central com 2 pontos: 
x = np.linspace(xbar-h, xbar+h)
plt.plot(x, f(x), label="f(x)")

x2 = np.array([ xbar + h,  xbar - h])
c_1_2 = fdcoeffV(1, xbar, x2) # approx for 1st derivative, 2 points
x3 = np.array([ xbar + h, xbar,  xbar - h])
c_2_2 = fdcoeffV(2, xbar, x3) # approx for 2nd derivative, 2 points



# (ii) central com 4 pontos: 
x4 = [xbar + h, xbar + h/2, xbar - h/2, xbar -h]
c_1_4 = fdcoeffV(1, xbar, x4) # approx for 1st derivative, 4 points
c_2_4 = fdcoeffV(2, xbar, x4) # approx for 2nd derivative, 4 points

p_f1p = plt.plot(x, f_1p(x), "--", label="f'(x)")
a_f1p2 = u_approx( x2, c_1_2, f)
a_f1p4 = u_approx( x4, c_1_4, f)
print(f"f2(xbar) {f_1p(xbar)}\n \
        u2f2(xbar) {a_f1p2}\n \
        u4f2(xbar) {a_f1p4}\n \
      ")
plt.plot(xbar, a_f1p2, "o", color=p_f1p[0].get_color(), label="1st derivative, 2 points")
plt.plot(xbar, a_f1p4, "*", color=p_f1p[0].get_color(), label="1st derivative, 4 points")
plt.plot(xbar, a_f1p4, "*", color=p_f1p[0].get_color(), label="1st derivative, 4 points")

p_f2p = plt.plot(x, f_2p(x), "-.", label="f''(x)")
a_f2p2 = u_approx(x3, c_2_2, f)
a_f2p4 = u_approx(x4, c_2_4, f)
print(f"f2(xbar) {f_2p(xbar)}\n \
        u2f2(xbar) {a_f2p2}\n \
        u4f2(xbar) {a_f2p4}\n\n \
      ")
plt.plot(xbar, a_f2p2, "o", color=p_f2p[0].get_color(), label="2nd derivative, 2 points")
plt.plot(xbar, a_f2p4, "*", color=p_f2p[0].get_color(), label="2nd derivative, 4 points")

plt.title(f"f(x) = exp(x)sin(x), h = {h}")
plt.legend()
plt.savefig('q4.png')
plt.show()



