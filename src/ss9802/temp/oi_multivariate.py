import numpy as np
from OceanPy.interpolation import polyfit2d, OI

import matplotlib.pyplot as plt

# define grid
nx, ny = 21, 21
lx, ly = 1, 1
xi, dx = np.linspace(-1, lx, nx, retstep=True)
yi, dy = np.linspace(-1, ly, ny, retstep=True)
# dx, dy = Lx / (nx - 1), Ly / (ny - 1)
xx, yy = np.meshgrid(xi, yi)
xc, yc = xi[0] + (nx - 1) * dx / 2, yi[0] + (ny - 1) * dy / 2

# create background field
nvar = 3
p_b = -xx*np.exp(-(2*xx)**2-(2.5*yy)**2)
u_b, v_b = np.gradient(p_b, dx)
u_b = -u_b

# make synthetic observations
nobs = 50
err = 0.2
var = np.random.uniform(-err, err, nobs)
x = np.random.uniform(-1, 1, nobs)
y = np.random.uniform(-1, 1, nobs)
p_o = - x * np.exp(-(2 * (x+var))**2 - (2.5 * (y+var))**2)





# var = [p, u, v]
n = nx * ny #* 3
p = len(p_o)

# background error covariance matrix
varian_b = 1
Lx = lx / 6
def Bmatrix(varian_b, Lx, Ly=None):
    B = np.matlib.ones((n, n))
    for m in range(1, n):
        mj = int(m / nx)
        mi = m - mj * nx

        xm = xc + (mi - int(nx / 2)) * dx
        ym = yc + (mj - int(ny / 2)) * dy

        for l in range(0, m):
            lj = int(l / nx)
            li = l - lj * nx

            xl = xc + (li - int(nx / 2)) * dx
            yl = yc + (lj - int(ny / 2)) * dy

            dist2 = (xm - xl) ** 2 + (ym - yl) ** 2
            cov = np.exp(-dist2 / (2 * Lx ** 2)) if Ly is None else np.exp(-dist2 / (Lx ** 2 + Ly ** 2))

            B[m, l] = cov
            B[l, m] = cov

    # variance background field
    for m in range(0, n):
        B[m, m] = varian_b

    return B
B = Bmatrix(varian_b, Lx)

fig, ax = plt.subplots()
BB = np.reshape(np.asarray(B), (nx, ny, nx, ny))
BBi = np.ma.masked_invalid(BB[int(np.floor(nx/2)), int(np.floor(ny/2)), :, :])
conf = ax.contourf(xx, yy, BBi)
ax.set_xlim([-np.sqrt(2)*Lx, np.sqrt(2)*Lx])
ax.set_ylim([-np.sqrt(2)*Lx, np.sqrt(2)*Lx])
plt.colorbar(conf)


# observation error covariance matrix
varian_r = np.var(p_o)
R = np.identity(p)
# R = varian_r * R

# OBSERVATION VECTOR
y_o = np.matrix(p_o).T
# y_o = np.matlib.empty((P, 1))
# y_o[0::2] = p_o.reshape((p, 1))#u_o
# y_o[1::2] = p_o.reshape((p, 1))

def Hmatrix():
    H = np.matlib.zeros((p, n))
    for k in range(p):

        # x, y distance from origin (ll-corner of grid) to observation
        xo = int(nx / 2) - np.ceil(xc / dx) + x[k] / dx
        yo = int(ny / 2) - np.ceil(yc / dy) + y[k] / dy

        i, j = int(xo), int(yo)

        if 0 <= i <= nx - 1 and 0 <= j <= ny - 1:

            wx = xo - i
            wy = yo - j

            i = i - 1 if i == nx - 1 else i
            j = j - 1 if j == ny - 1 else j

            H[k, j * nx + i] = (1 - wx) * (1 - wy)
            H[k, j * nx + i + 1] = wx * (1 - wy)
            H[k, j * nx + nx + i] = (1 - wx) * wy
            H[k, j * nx + nx + i + 1] = wx * wy

            # print('Check sum: %s' % (wx * (1 - wy) + (1 - wx) * (1 - wy) + wx * wy + (1 - wx) * wy))

        else:
            raise ValueError('Observation point (%s, %s) is not within grid domain.' % (x[k], y[k]))
    return H
H = Hmatrix()

# background field vector at grid points
x_b = np.reshape(p_b, (n, 1))

# background field vector
y_b = H * x_b

# convert=True
# if convert:
#     conv = np.diag(list(map(lambda x: 4*sigma*x**3, np.asarray(y_b)[:,0])))
#     H = conv * H
#
#     y_b = np.reshape(list(map(lambda x: sigma*x**4, np.asarray(y_b)[:, 0])), (P, 1))

# weight matrix
W = B * H.T * (R + H * B * H.T).I

# analysis field vector
x_a = x_b + W * (y_o - y_b)

p_a = np.asarray(np.reshape(x_a, (ny, nx)))

vmin, vmax = -0.25, 0.25
fig, ax = plt.subplots(2, 1)
pcol = ax[0].pcolor(xx, yy, p_b, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
ax[0].scatter(x, y, c=p_o[:p], cmap=plt.cm.seismic, vmin=vmin, vmax=vmax, edgecolors='k')
plt.colorbar(pcol, ax=ax[0])
ax[0].quiver(xx, yy, u_b, v_b)
ax[0].set_title('Background field')

pcol = ax[1].pcolor(xx, yy, p_a, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
ax[1].scatter(x, y, c=p_o[:p], cmap=plt.cm.seismic, vmin=vmin, vmax=vmax, edgecolors='k')
plt.colorbar(pcol, ax=ax[1])
ax[1].set_title('Analysis field')

