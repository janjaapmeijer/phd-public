# http://twister.caps.ou.edu/OBAN2002/OI.pdf
# assim.chocolate.fr/ana2d.py.txt
# http://stockage.univ-brest.fr/~herbette/Data-Analysis/data-analysis-opt-interp.pdf


import matplotlib.pyplot as plt
from OceanPy.interpolation import polyfit2d
import numpy as np


lon = np.array([1, -0.5, -0.5, 0.5, -1])
lat = np.array([-0.5, 1, -0.5, 0.75, -1])
u_o = 3*lon**2 + 2*lat**3 + 2
v_o = -2*lon**3 - 3*lat**2 + 3
nvar = 2

# define grid of background field
nx, ny = 20, 20

# define grid
# dx, dy = Lx / (nx - 1), Ly / (ny - 1)
xi, dx, = np.linspace(-1, 1, nx, retstep=True)
yi, dy = np.linspace(-1, 1, ny, retstep=True)
xx, yy = np.meshgrid(xi, yi)
xc, yc = xi[0] + (nx - 1) * dx / 2, yi[0] + (ny - 1) * dy / 2

f = polyfit2d(lon, lat, u_o, order=1)
u_b = f(xx, yy)
f = polyfit2d(lon, lat, v_o, order=1)
v_b = f(xx, yy)

N = nx * ny# * nvar #len(grid_points) * len(vars)
Nn = nx * ny * nvar
P = len(u_o) #len(obs)
Pn = len(u_o) * nvar

# BACKGROUND ERROR COVARIANCE MATRIX
# Gaussian fuction to model the correlation between analysis point i and analysis point j
# gamma_ij = np.exp(-(r_ij/L)**2)
# r_ij is the distance between i and j
# L length scale, in the ocean mesoscale processes have a length scale on the order of the radius of deformation

Lx=0.5
Ly = 0.5
varian_b = 1#np.var(T_b)

B = np.matrix(np.zeros((Nn, Nn)))
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

        dist2 = (xm - xl)**2 + (ym - yl)**2
        # cov = np.exp(-dist2 / (2 * L**2))
        cov = np.exp(-dist2 / (Lx**2 + Ly**2))

        for ivar in range(nvar):
            B[nvar * m + ivar, nvar * l + ivar] = cov
            B[nvar * l + ivar, nvar * m + ivar] = cov

# variance background field
for m in range(0, Nn):
    B[m, m] = varian_b

# OBSERVATION ERROR COVARIANCE MATRIX
varian_r = np.var(u_o) # (Kalnay 5.4.37)
R = np.identity(Pn)
R = varian_r * R


# OBSERVATION VECTOR
y_o = np.matlib.empty((Pn, 1))
y_o[0::2] = u_o.reshape((P, 1))#u_o
y_o[1::2] = v_o.reshape((P, 1))

# FORWARD OPERATOR OR OBSERVATION OPERATOR MATRIX
H = np.matrix(np.zeros((Pn, Nn)))
for k in range(P):
    # print(k)
    # x, y distance from origin (ll-corner of grid) to observation
    xo = int(nx / 2) - np.ceil(xc / dx) + lon[k] / dx
    yo = int(ny / 2) - np.ceil(yc / dy) + lat[k] / dy

    i, j = int(xo), int(yo)

    if 0 <= i <= nx - 1 and 0 <= j <= ny - 1:

        i = i - 1 if i == nx - 1 else i
        j = j - 1 if j == ny - 1 else j

        wx = xo - i
        wy = yo - j

        for ivar in range(nvar):
            # print(nvar * k + ivar)
            H[nvar * k + ivar, nvar * (j*nx + i) + ivar] = (1 - wx) * (1 - wy)
            H[nvar * k + ivar, nvar * (j*nx + i + 1) + ivar] = wx * (1 - wy)
            H[nvar * k + ivar, nvar * (j*nx + nx + i) + ivar] = (1 - wx) * wy
            H[nvar * k + ivar, nvar * (j*nx + nx + i + 1) + ivar] = wx * wy

    else:
        raise ValueError('Observation point (%s, %s) is not within grid domain.' % (lon[k], lat[k]))

x_b = np.matlib.empty((Nn, 1))
x_b[0::2] = np.reshape(u_b, (N, 1))
x_b[1::2] = np.reshape(v_b, (N, 1))

y_b = H * x_b

# print(x_b, y_o, y_b, B, H)

d = y_o - y_b

# %%  Construct the gain matrix W = BH'[R+HBH']^(-1)
W = B * H.T * (R + H * B * H.T).I

# %%  Compute the analysed fields.
x_a = x_b + W * d

# %%  Analysis error covariance matrix A = (1-WH)B
I = np.identity(Nn)
A = (I - W * H) * B

u_a = np.reshape(x_a[0::2], (ny, nx))
v_a = np.reshape(x_a[1::2], (ny, nx))


### PLOT
plt.figure()
vmin, vmax = u_b.min(), u_b.max()
plt.contourf(xx, yy, u_b, np.linspace(vmin, vmax, 13))
plt.colorbar()
# plt.scatter(lon, lat, c=yb[0,0], s=400, vmin=vmin, vmax=vmax)
# plt.scatter(lon, lat, c=T_ointerp, s=400, vmin=vmin, vmax=vmax)
scat = plt.scatter(lon, lat, c=u_o, s=100, vmin=vmin, vmax=vmax)
plt.axis('equal')
plt.colorbar(scat)
plt.title('Background field u')

plt.figure()
plt.contourf(xx, yy, np.asarray(u_a), np.linspace(vmin, vmax, 13))
plt.colorbar()
scat = plt.scatter(lon, lat, c=u_o, s=100, vmin=vmin, vmax=vmax)
plt.colorbar(scat)
plt.title('Analysis field')

### PLOT
plt.figure()
vmin, vmax = v_b.min(), v_b.max()
plt.contourf(xx, yy, v_b, np.linspace(vmin, vmax, 13))
plt.colorbar()
# plt.scatter(lon, lat, c=yb[0,0], s=400, vmin=vmin, vmax=vmax)
# plt.scatter(lon, lat, c=T_ointerp, s=400, vmin=vmin, vmax=vmax)
scat = plt.scatter(lon, lat, c=v_o, s=100, vmin=vmin, vmax=vmax)
plt.axis('equal')
plt.colorbar(scat)
plt.title('Background field v')

plt.figure()
plt.contourf(xx, yy, np.asarray(v_a), np.linspace(vmin, vmax, 13))
plt.colorbar()
scat = plt.scatter(lon, lat, c=v_o, s=100, vmin=vmin, vmax=vmax)
plt.colorbar(scat)
plt.title('Analysis field')


# x, dx = np.linspace(-1, 1, 21, retstep=True)
# y, dy = np.linspace(-1, 1, 21, retstep=True)
#
# xx, yy = np.meshgrid(x, y)
#
# rlim= 1e-3
# nr = 50
# var = np.random.uniform(-rlim, rlim, nr)
# x_o = np.random.uniform(-1, 1, nr)
# y_o = np.random.uniform(-1, 1, nr)
# p_o = - y_o * np.exp(-2 * x_o**2 - 2.5 * y_o**2) + var
#
# p = -yy*np.exp(-(2*xx)**2-(2.5*yy)**2)
#
# u, v = np.gradient(p, dx)
# u = -u
#
# plt.figure()
# plt.pcolor(xx, yy, p, cmap=plt.cm.seismic, vmin=-0.25, vmax=0.25)
# plt.scatter(x_o, y_o, c=p_o, cmap=plt.cm.seismic, vmin=-0.25, vmax=0.25, edgecolors='k')
# plt.colorbar()
# plt.quiver(xx, yy, u, v)