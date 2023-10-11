import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class Edge(Enum):
    TOP = 1
    BOTTOM = 2
    LEFT = 3
    RIGHT = 4


class Vertex(Enum):
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4


class Region(Enum):
    EDGE = 1
    VERTEX = 2
    INSIDE = 3


## Input data
# Geometry and material properties
Lix = -1
Liy = -1
Lfx = 1
Lfy = 1
lamb = 0.01

# Final time
tend = 100

## Discretization
Nx = 17  # spatial x
Ny = 17  # spatial y
Nt = 1001  # temporal
x = np.linspace(Lix, Lfx, Nx)
y = np.linspace(Liy, Lfy, Ny)
t = np.linspace(0, tend, Nt)
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = t[1] - t[0]

beta1 = lamb * dt / (dx**2)
beta2 = lamb * dt / (dy**2)

# Initial conditions
T0 = np.zeros((Nx, Ny))
for i, xi in enumerate(x):
    for j, yj in enumerate(y):
        T0[i, j] = 10 * np.exp((-20) * (xi**2 + yj**2))

# Flatten important matrices
[xg, yg] = np.meshgrid(x, y)
x = xg.flatten()
y = yg.flatten()
T0 = T0.flatten()

# regions
region_B = np.arange(0, Nx, 1)
region_R = np.arange(Nx - 1, Nx * Ny, Nx)
region_T = np.arange(Nx * (Ny - 1), Nx * Ny, 1)
region_L = np.arange(0, Nx * (Ny - 1) + 1, Nx)


def check_region(i):
    in_edge_LEFT = i in set(region_L)
    in_edge_RIGHT = i in set(region_R)
    in_edge_BOTTOM = i in set(region_B)
    in_edge_TOP = i in set(region_T)
    in_edge = in_edge_LEFT or in_edge_RIGHT or in_edge_BOTTOM or in_edge_TOP

    region = Region.INSIDE
    if in_edge:  # if it is not in at least one edge, it is inside
        in_vertex_LB = in_edge_LEFT and in_edge_BOTTOM
        in_vertex_RB = in_edge_RIGHT and in_edge_BOTTOM
        in_vertex_LT = in_edge_LEFT and in_edge_TOP
        in_vertex_RT = in_edge_RIGHT and in_edge_TOP
        in_vertex = in_vertex_LB or in_vertex_RB or in_vertex_LT or in_vertex_RT

        if in_vertex:  # it is in two edges simultaneously
            if in_vertex_LB:
                region = Vertex.BOTTOM_LEFT
            elif in_vertex_LT:
                region = Vertex.TOP_LEFT
            elif in_vertex_RB:
                region = Vertex.BOTTOM_RIGHT
            elif in_vertex_RT:
                region = Vertex.TOP_RIGHT
        else:  # only in one edge
            if in_edge_TOP:
                region = Edge.TOP
            elif in_edge_BOTTOM:
                region = Edge.BOTTOM
            elif in_edge_RIGHT:
                region = Edge.RIGHT
            elif in_edge_LEFT:
                region = Edge.LEFT

    return region


def vertices(i, k, T, region):
    if region == Vertex.BOTTOM_LEFT:
        T[i, k] = (
            beta1 * (2 * T[i + 1, k - 1] - 2 * T[i, k - 1])
            + beta2 * (2 * T[i + Nx, k - 1] - 2 * T[i, k - 1])
            + T[i, k - 1]
        )
    elif region == Vertex.TOP_LEFT:
        T[i, k] = (
            beta1 * (2 * T[i + 1, k - 1] - 2 * T[i, k - 1])
            + beta2 * (2 * T[i - Nx, k - 1] - 2 * T[i, k - 1])
            + T[i, k - 1]
        )
    elif region == Vertex.BOTTOM_RIGHT:
        T[i, k] = (
            beta1 * (2 * T[i - 1, k - 1] - 2 * T[i, k - 1])
            + beta2 * (2 * T[i + Nx, k - 1] - 2 * T[i, k - 1])
            + T[i, k - 1]
        )
    elif region == Vertex.TOP_RIGHT:
        T[i, k] = (
            beta1 * (2 * T[i - 1, k - 1] - 2 * T[i, k - 1])
            + beta2 * (2 * T[i - Nx, k - 1] - 2 * T[i, k - 1])
            + T[i, k - 1]
        )


def edges(i, k, T, region):
    # Edges without vertices
    if region == Edge.LEFT:
        T[i, k] = (
            beta1 * (2 * T[i + 1, k - 1] - 2 * T[i, k - 1])
            + beta2 * (T[i - Nx, k - 1] - 2 * T[i, k - 1] + T[i + Nx, k - 1])
            + T[i, k - 1]
        )
    elif region == Edge.RIGHT:
        T[i, k] = (
            beta1 * (2 * T[i - 1, k - 1] - 2 * T[i, k - 1])
            + beta2 * (T[i - Nx, k - 1] - 2 * T[i, k - 1] + T[i + Nx, k - 1])
            + T[i, k - 1]
        )
    elif region == Edge.BOTTOM:
        T[i, k] = (
            beta1 * (T[i - 1, k - 1] - 2 * T[i, k - 1] + T[i + 1, k - 1])
            + beta2 * (2 * T[i + Nx, k - 1] - 2 * T[i, k - 1])
            + T[i, k - 1]
        )
    elif region == Edge.TOP:
        T[i, k] = (
            beta1 * (T[i - 1, k - 1] - 2 * T[i, k - 1] + T[i + 1, k - 1])
            + beta2 * (2 * T[i - Nx, k - 1] - 2 * T[i, k - 1])
            + T[i, k - 1]
        )


T = np.zeros((Nx * Ny, Nt))
for k, tk in enumerate(t):
    if tk == 0:
        for i, xi in enumerate(x):
            T[i, k] = T0[i]
    else:
        for i, xi in enumerate(x):
            region = check_region(i)

            if region == Region.INSIDE:
                T[i, k] = (
                    beta1 * (T[i - 1, k - 1] - 2 * T[i, k - 1] + T[i + 1, k - 1])
                    + beta2 * (T[i - Nx, k - 1] - 2 * T[i, k - 1] + T[i + Nx, k - 1])
                    + T[i, k - 1]
                )
            elif region in Vertex:
                vertices(i, k, T, region)
            elif region in Edge:
                edges(i, k, T, region)

Tmatrix = np.zeros((Nt, Ny, Nx))
for k in range(Nt):
    Tmatrix[k, :, :] = np.reshape(T[:, k], (Ny, Nx))


plt.figure()
plt.contourf(xg, yg, Tmatrix[0, :, :], 20, cmap="hot")
plt.colorbar()
plt.title("t = 0")
plt.savefig("figs/q7a_heatmap_t0.png", dpi=300)

plt.figure()
plt.contourf(xg, yg, Tmatrix[250, :, :], 20, cmap="hot")
plt.colorbar()
plt.title("t = 25")
plt.savefig("figs/q7a_heatmap_t25.png", dpi=300)

plt.figure()
plt.contourf(xg, yg, Tmatrix[500, :, :], 20, cmap="hot")
plt.colorbar()
plt.title("t = 50")
plt.savefig("figs/q7a_heatmap_t50.png", dpi=300)

plt.figure()
plt.contourf(xg, yg, Tmatrix[750, :, :], 20, cmap="hot")
plt.colorbar()
plt.title("t = 75")
plt.savefig("figs/q7a_heatmap_t75.png", dpi=300)

plt.figure()
plt.contourf(xg, yg, Tmatrix[1000, :, :], 20, cmap="hot")
plt.colorbar()
plt.title("t = 100")
plt.savefig("figs/q7a_heatmap_t100.png", dpi=300)

# Plotting the solution
xplot = np.linspace(Lix, Lfx, Nx)
yplot = np.linspace(Liy, Lfy, Ny)
plt.figure()
for k, tk in enumerate(t):
    # if tk == 0:
    #     plt.plot(xplot,Tmatrix[k,:,((Ny-1)//2)],'r')
    if tk == 25:
        plt.plot(xplot, Tmatrix[k, :, ((Ny - 1) // 2)], "darkorange")
    elif tk == 50:
        plt.plot(xplot, Tmatrix[k, :, ((Ny - 1) // 2)], "y")
    elif tk == 75:
        plt.plot(xplot, Tmatrix[k, :, ((Ny - 1) // 2)], "g")
    elif tk == 100:
        plt.plot(xplot, Tmatrix[k, :, ((Ny - 1) // 2)], "blue")
    else:
        continue
plt.title("Temperature profile (y=0)")
plt.xlabel("x")
plt.ylabel("Temperature")
plt.savefig("figs/q7a_temperature_profile.png", dpi=300)
