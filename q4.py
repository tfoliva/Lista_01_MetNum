import numpy as np
import matplotlib.pyplot as plt
from fdcoeff import fdcoeffV, approx_derivative, u_approx
import time

start = time.time()


# Questão 04
# considere f(x) = exp(x) sin(x)
def f(x):
    return np.exp(x) * np.sin(x)


def f_1p(x):
    return np.exp(x) * np.sin(x) + np.exp(x) * np.cos(x)


def f_2p(x):
    return 2 * np.exp(x) * np.cos(x)


# encontre em x = 2
h = 1
xbar = 2

# (i) central com 2 pontos:
x = np.linspace(xbar - h, xbar + h)
plt.plot(x, f(x), label="f(x)")

x2 = np.array([xbar + h, xbar - h])
c_1_2 = fdcoeffV(1, xbar, x2)  # approx for 1st derivative, 2 points
x3 = np.array([xbar + h, xbar, xbar - h])
c_2_3 = fdcoeffV(2, xbar, x3)  # approx for 2nd derivative, 3 points


# (ii) central com 4 pontos:
x4 = [xbar + h, xbar + h / 2, xbar - h / 2, xbar - h]
c_1_4 = fdcoeffV(1, xbar, x4)  # approx for 1st derivative, 4 points
x5 = [xbar + h, xbar + h / 2, xbar, xbar - h / 2, xbar - h]
c_2_5 = fdcoeffV(2, xbar, x5)  # approx for 2nd derivative, 5 points

p_f1p = plt.plot(x, f_1p(x), "--", label="f'(x)")
a_f1p2 = u_approx(x2, c_1_2, f)
a_f1p4 = u_approx(x4, c_1_4, f)
print(
    f"f2(xbar) {f_1p(xbar)}\n \
        u2f2(xbar) {a_f1p2}\n \
        u4f2(xbar) {a_f1p4}\n \
      "
)
plt.plot(
    xbar, a_f1p2, "o", color=p_f1p[0].get_color(), label="1st derivative, 2 points"
)
plt.plot(
    xbar, a_f1p4, "*", color=p_f1p[0].get_color(), label="1st derivative, 4 points"
)

p_f2p = plt.plot(x, f_2p(x), "-.", label="f''(x)")
a_f2p3 = u_approx(x3, c_2_3, f)
a_f2p5 = u_approx(x5, c_2_5, f)
print(
    f"f2(xbar) {f_2p(xbar)}\n \
        u2f2(xbar) {a_f2p3}\n \
        u4f2(xbar) {a_f2p5}\n\n \
      "
)
plt.plot(
    xbar, a_f2p3, "o", color=p_f2p[0].get_color(), label="2nd derivative, 3 points"
)
plt.plot(
    xbar, a_f2p5, "*", color=p_f2p[0].get_color(), label="2nd derivative, 5 points"
)

plt.title(f"f(x) = exp(x)sin(x), h = {h}")
plt.legend()
plt.savefig("figs/q4.png", dpi=300)

# sensibility to h
h_array = np.linspace(0.1, 1)
d1_2 = np.zeros(len(h_array))
d1_4 = np.zeros(len(h_array))
d2_3 = np.zeros(len(h_array))
d2_5 = np.zeros(len(h_array))
e1_2 = np.zeros(len(h_array))
e1_4 = np.zeros(len(h_array))
e2_3 = np.zeros(len(h_array))
e2_5 = np.zeros(len(h_array))

for i, h in enumerate(h_array):
    x2 = [xbar - h, xbar + h]
    x3 = [xbar - h, xbar, xbar + h]
    x4 = [xbar - h, xbar - h / 2, xbar + h / 2, xbar + h]
    x5 = [xbar - h, xbar - h / 2, xbar, xbar + h / 2, xbar + h]
    d1_2[i] = approx_derivative(1, xbar, x2, f)
    d1_4[i] = approx_derivative(1, xbar, x4, f)
    d2_3[i] = approx_derivative(2, xbar, x3, f)
    d2_5[i] = approx_derivative(2, xbar, x5, f)

    e1_2[i] = (d1_2[i] - f_1p(xbar)) ** 2
    e1_4[i] = (d1_4[i] - f_1p(xbar)) ** 2
    e2_3[i] = (d2_3[i] - f_2p(xbar)) ** 2
    e2_5[i] = (d2_5[i] - f_2p(xbar)) ** 2
end = time.time()

plt.figure()
plt.title("Finite Differences approximations")
plt.plot(h_array, d1_2, label="d1_2")
plt.plot(h_array, d1_4, label="d1_4")
plt.plot(h_array, d2_3, label="d2_3")
plt.plot(h_array, d2_5, label="d2_4")
plt.xlabel("h")
plt.legend()

plt.figure()
fst_der = plt.loglog(h_array, e1_2, label="e1_2")
plt.loglog(h_array, e1_4, "--", label="e1_4", color=fst_der[0].get_color())
scd_der = plt.loglog(h_array, e2_3, label="e2_3")
plt.loglog(h_array, e2_5, "--", label="e2_5", color=scd_der[0].get_color())
plt.xlabel("h")
plt.legend()
plt.grid(which="both", color="gray", linestyle="--", linewidth=0.5)
plt.savefig("figs/q4_error.png", dpi=300)
plt.show()

print(f"Q4 took {end-start} seconds")
