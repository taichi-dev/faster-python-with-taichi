# code from https://github.com/rougier/numpy-book#
import numpy as np
import numba
import matplotlib.pyplot as plt

n = 400
Du, Dv, F, k = 0.16, 0.08, 0.035, 0.065

Z = np.zeros((n+2, n+2), [('U', np.double),
                          ('V', np.double)])
U, V = Z['U'], Z['V']
u, v = U[1:-1, 1:-1], V[1:-1, 1:-1]

r = 20
u[...] = 1.0
U[n//2-r:n//2+r, n//2-r:n//2+r] = 0.50
V[n//2-r:n//2+r, n//2-r:n//2+r] = 0.25
u += 0.05*np.random.uniform(-1, +1, (n, n))
v += 0.05*np.random.uniform(-1, +1, (n, n))

@numba.jit(nopython=True)
def update(u, v, U, V):
    for i in range(10):
        Lu = (                  U[0:-2, 1:-1] +
              U[1:-1, 0:-2] - 4*U[1:-1, 1:-1] + U[1:-1, 2:] +
                                U[2:  , 1:-1])
        Lv = (                  V[0:-2, 1:-1] +
              V[1:-1, 0:-2] - 4*V[1:-1, 1:-1] + V[1:-1, 2:] +
                                V[2:  , 1:-1])
        uvv = u*v*v
        u += (Du*Lu - uvv + F*(1 - u))
        v += (Dv*Lv + uvv - (F + k) * v)


fig = plt.figure(figsize=(4, 4))
fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
im = plt.imshow(V, interpolation='bilinear', cmap=plt.cm.viridis)
plt.xticks([])
plt.yticks([])
for _ in range(2000):
    for _ in range(10):
        update(u, v, U, V)

    im.set_data(V)
    im.set_clim(vmin=V.min(), vmax=V.max())
    plt.pause(0.001)
