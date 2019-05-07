import numpy as np
from OceanPy.interpolation import polyfit2d, OI

import matplotlib.pyplot as plt

# define grid
nx, ny = 21, 21
lx, ly = 1e7, 1e7
# dx, dy = lx / (nx - 1), ly / (ny - 1)
# dxdy = dx * dy
xi, dx = np.linspace(0, lx, nx, retstep=True)
yi, dy = np.linspace(0, ly, ny, retstep=True)
xx, yy = np.meshgrid(xi, yi)
xc, yc = xi[0] + (nx - 1) * dx / 2, yi[0] + (ny - 1) * dy / 2

# create background field
f = polyfit2d([xx[0, 0], xx[0, -1], xx[-1, 0], xx[-1, -1]],
              [yy[0, 0], yy[0, -1], yy[-1, 0], yy[-1, -1]],
              [-10, -15, -40, -45], order=1)
t_b = f(xx, yy)

N = nx * ny
# P = len(obs_fld)
P = 3

# background error covariance matrix
varian_b = 1
Lx = lx / 6
def Bmatrix(varian_b, Lx, Ly=None):
    B = np.matlib.ones((N, N))
    for m in range(1, N):
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
    for m in range(0, N):
        B[m, m] = varian_b

    return B
B = Bmatrix(varian_b, Lx)

# make synthetic radiances obsservations
nxobs = np.array([np.floor((nx-1)/4)+1, np.floor((nx-1)/4)+1, np.floor(3*(nx-1)/4)+1, np.floor(nx/2)+1])
nyobs = np.array([np.floor((ny-1)/4)+1, np.floor(3*(ny-1)/4)+1, np.floor(3*(ny-1)/4)+1, np.floor(nx/2)+1])
dxobs = np.array([0.3, 0.5, 0.7, 0.0])
dyobs = np.array([0.3, 0.5, 0.7, 0.99])
tobs = np.array([-20, -30, -40, -50])

# observational field vector
y_o = np.zeros((P, 1))
x, y = np.zeros(P), np.zeros(P)
sigma = 4.567E-8
for nobs in range(P):
    x[nobs] = (nxobs[nobs] - 1 + dxobs[nobs]) * dx
    y[nobs] = (nyobs[nobs] - 1 + dyobs[nobs]) * dy
    y_o[nobs] = sigma * (tobs[nobs] + 273)**4
    # y_o[nobs] = tobs[nobs]


# observation error covariance matrix
teps = 1.5
varian_r = sigma * ((220 + teps)**4 - 220**4)
R = varian_r * np.matrix(np.ones((P, P)))
# varian_r = np.var(y_o)
# R = varian_r * np.identity(P)

def Hmatrix():
    H = np.matlib.zeros((P, N))
    for k in range(P):
        xo = int(nx / 2) - np.ceil(xc / dx) + x[k] / dx
        yo = int(ny / 2) - np.ceil(yc / dy) + y[k] / dy

        i, j = int(xo), int(yo)
        if 0 <= i <= nx - 1 and 0 <= j <= ny - 1:

            wx = xo - i
            wy = yo - j

            # i = i - 1 if i == nx - 1 else i
            # j = j - 1 if j == ny - 1 else j

            H[k, j * nx + i] = (1 - wx) * (1 - wy)
            H[k, j * nx + i + 1] = wx * (1 - wy)
            H[k, j * nx + nx + i] = (1 - wx) * wy
            H[k, j * nx + nx + i + 1] = wx * wy

            print('Check sum: %s' % (wx * (1 - wy) + (1 - wx) * (1 - wy) + wx * wy + (1 - wx) * wy))


        else:
            raise ValueError('Observation point (%s, %s) is not within grid domain.' % (x[k], y[k]))
    return H
H = Hmatrix()

# background field vector at grid points
x_b = np.reshape(t_b, (N, 1)) + 273

# background field vector
y_b = H * x_b

convert=True
if convert:
    conv = np.diag(list(map(lambda x: 4*sigma*x**3, np.asarray(y_b)[:,0])))
    H = conv * H

    y_b = np.reshape(list(map(lambda x: sigma*x**4, np.asarray(y_b)[:, 0])), (P, 1))

# weight matrix
W = B * H.T * (R + H * B * H.T).I

# analysis field vector
x_a = x_b + W * (y_o - y_b)

t_a = np.asarray(np.reshape(x_a - 273, (nx, ny)))
# t_a = np.asarray(np.reshape(x_a, (nx, ny)))


plt.figure()
plt.contourf(xx, yy, t_b, np.linspace(-50, -10, 9))
plt.colorbar()
scat = plt.scatter(x, y, c=tobs[:P], vmin=-50, vmax=-10, s=100)
plt.colorbar(scat)

plt.figure()
plt.contourf(xx, yy, np.asarray(t_a), np.linspace(-50, -10, 9))
plt.colorbar()
scat = plt.scatter(x, y, c=tobs[:P], vmin=-50, vmax=-10, s=100)
plt.colorbar(scat)
plt.title('Analysis field')