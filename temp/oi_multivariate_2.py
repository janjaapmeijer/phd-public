import numpy as np
from OceanPy.interpolation import polyfit2d, OI

import matplotlib.pyplot as plt

# define grid
nx, ny = 21, 21
lx, ly = 1, 1
xi, dx = np.linspace(-lx, lx, nx, retstep=True)
yi, dy = np.linspace(-ly, ly, ny, retstep=True)
# dx, dy = Lx / (nx - 1), Ly / (ny - 1)
xx, yy = np.meshgrid(xi, yi)
xc, yc = xi[0] + (nx - 1) * dx / 2, yi[0] + (ny - 1) * dy / 2

# create background field
nvar = 3
p_b = -xx*np.exp(-(2*xx)**2-(3*yy)**2)
u_b, v_b = np.gradient(p_b, dx)
u_b = -u_b


# make synthetic observations
nobs = 50
err = 0.5
var = np.random.uniform(-err, err, nobs)
x = np.random.uniform(-lx, lx, nobs)
y = np.random.uniform(-ly, ly, nobs)
p_o = - y * np.exp(-(2 * (x+var))**2 - (3 * (y+var))**2)


n = nx * ny #* 3
N = n * nvar
p = len(p_o)


# # background error covariance matrix
# varian_b = 1
# Lx = lx/6
# def Bmatrix(varian_b, Lx, Ly=None):
#     B = np.matlib.ones((N, N))
#     for k in range(1, n):
#         kj = int(k / nx)
#         ki = k - kj * nx
#
#         xk = xc + (ki - int(nx / 2)) * dx
#         yk = yc + (kj - int(ny / 2)) * dy
#
#         for l in range(0, k):
#             lj = int(l / nx)
#             li = l - lj * nx
#
#             xl = xc + (li - int(nx / 2)) * dx
#             yl = yc + (lj - int(ny / 2)) * dy
#
#             dist2 = (xl - xk) ** 2 + (yl - yk) ** 2
#             cov = np.exp(-dist2 / (2 * Lx ** 2)) if Ly is None else np.exp(-dist2 / (Lx ** 2 + Ly ** 2))
#
#             for ivar in range(nvar):
#                 for jvar in range(nvar):
#                     # Cpp
#                     if ivar == jvar == 0:
#                         B[nvar * k + ivar, nvar * l + jvar] = cov
#                         B[nvar * l + ivar, nvar * k + jvar] = cov
#                     # Cpu and Cup
#                     # if (ivar == 0 and jvar == 1) or (ivar == 1 and jvar == 0):
#                     #     # if jvar == 0:
#                     #     #     cov = -cov
#                     #     B[nvar * k + ivar, nvar * l + jvar] = (B[nvar * k, nvar * l] - B[nvar * k, nvar * l - 1]) / (yl - yk) if (yl - yk) != 0  else 0 # cov / (f * (yl - yk)) if (yl - yk) != 0 else 0#(covy - covym) / f
#                     #     B[nvar * l + ivar, nvar * k + jvar] = - (B[nvar * k, nvar * l] - B[nvar * k, nvar * l - 1]) / (yl - yk) if (yl - yk) != 0  else 0# - cov / (f * (yl - yk)) if (yl - yk) != 0 else 0#-(covy - covym) / f
#                     # # # # Cpv and Cvp
#                     # # if (ivar == 0 and jvar == 2) or (ivar == 2 and jvar == 0):
#                     # #     B[nvar * k + ivar, nvar * l + jvar] = -covx / (f * (xl - xk)) if (xl - xk) != 0 else -covx / f
#                     # #     B[nvar * l + ivar, nvar * k + jvar] = covx / (f * (xl - xk)) if (xl - xk) != 0 else covx / f
#                     # # Cuu
#                     # if ivar == jvar == 1:
#                     #     B[nvar * k + ivar, nvar * l + jvar] = - (cov**2 / (f * (yl - yk))) / (f * (yl - yk)) if (yl - yk) != 0 else 0#-covy**2 / f**2
#                     #     B[nvar * l + ivar, nvar * k + jvar] = - (cov**2 / (f * (yl - yk))) / (f * (yl - yk)) if (yl - yk) != 0 else 0#-covy**2 / f**2
#                     # # Cvv
#                     # if ivar == jvar == 2:
#                     #     B[nvar * k + ivar, nvar * l + jvar] = - covx**2 / (f**2 * (xl - xk) ** 2) if (xl - xk) != 0 else 0# -covx**2 / f**2
#                     #     B[nvar * l + ivar, nvar * k + jvar] = - covx**2 / (f**2 * (xl - xk) ** 2) if (xl - xk) != 0 else 0#-covx**2 / f**2
#                     # # # Cuv and Cvu
#                     # # if (ivar == 1 and jvar == 2) or (ivar == 2 and jvar == 1):
#                     # #     B[nvar * k + ivar, nvar * l + jvar] = covx * covy / (f**2 * (xl - xk) * (yl - yk)) if (xl - xk) != 0 and (yl - yk) != 0 else covx * covy / f**2
#                     # #     B[nvar * l + ivar, nvar * k + jvar] = covx * covy / (f**2 * (xl - xk) * (yl - yk)) if (xl - xk) != 0 and (yl - yk) != 0 else covx * covy / f**2
#
#     # variance background field
#     for k in range(0, n):
#         B[k, k] = varian_b
#
#     return B
# B = Bmatrix(varian_b, Lx)
#
# f=1
#
#
# fig, ax = plt.subplots(nvar, nvar)
# for ivar in range(nvar):
#     for jvar in range(nvar):
#         BB = np.reshape(np.asarray(B)[ivar::nvar, jvar::nvar], (nx, ny, nx, ny))
#         BBi = np.ma.masked_invalid(BB[int(np.floor(nx/2)), int(np.floor(ny/2)), :, :])
#         conf = ax[ivar, jvar].contourf(xx, yy, BBi)
#         ax[ivar, jvar].set_xlim([-np.sqrt(2)*Lx, np.sqrt(2)*Lx])
#         ax[ivar, jvar].set_ylim([-np.sqrt(2)*Lx, np.sqrt(2)*Lx])
#         plt.colorbar(conf, ax=ax[ivar, jvar])
#
# fig, ax = plt.subplots(nvar, nvar)
# for ivar in range(nvar):
#     for jvar in range(nvar):
#         # Cpp
#         BB = np.reshape(np.asarray(B)[0::nvar, 0::nvar], (nx, ny, nx, ny))
#         # if ivar == jvar == 0:
#         # BBi = np.ma.masked_invalid(BB[int(np.floor(nx / 2)), int(np.floor(ny / 2)), :, :])
#         BBi = np.ma.masked_invalid(BB[0, 0, :, :])
#         # Cpu and Cup
#         if (ivar == 0 and jvar == 1) or (ivar == 1 and jvar == 0):
#             if ivar == 0:
#                 BBi = -BBi
#             BBi = (1 / f) * np.gradient(BBi, dy)[0]
#         # Cuu
#         if ivar == jvar == 1:
#             BBi = -(1 / f**2) * np.gradient(np.gradient(BBi, dy)[0], dy)[0]
#         # Cuv and Cvu
#         if (ivar == 1 and jvar == 2) or (ivar == 2 and jvar == 1):
#             BBi = (1 / f**2) * np.gradient(np.gradient(BBi, dx)[0], dy)[1]
#         # Cpv and Cvp
#         if (ivar == 0 and jvar == 2) or (ivar == 2 and jvar == 0):
#             if ivar == 0:
#                 BBi = -BBi
#             BBi = - (1 / f) * np.gradient(BBi, dx)[1]
#         # Cvv
#         if ivar == jvar == 2:
#             BBi = -(1 / f**2) * np.gradient(np.gradient(BBi, dx)[1], dx)[1]
#
#         conf = ax[ivar, jvar].contourf(xx, yy, BBi)
#         # ax[ivar, jvar].set_xlim([-np.sqrt(2)*Lx, np.sqrt(2)*Lx])
#         # ax[ivar, jvar].set_ylim([-np.sqrt(2)*Lx, np.sqrt(2)*Lx])
#         # ax[ivar, jvar].set_xlim([-0.5, 0.5])
#         # ax[ivar, jvar].set_ylim([-0.5, 0.5])
#         plt.colorbar(conf, ax=ax[ivar, jvar])

# background error covariance matrix
varian_b = 1
Lx = lx/6
def Bmatrix(varian_b, Lx, Ly=None):
    B = np.matlib.zeros((n, n))
    for k in range(1, n):
        kj = int(k / nx)
        ki = k - kj * nx

        xk = xc + (ki - int(nx / 2)) * dx
        yk = yc + (kj - int(ny / 2)) * dy

        for l in range(0, k):
            lj = int(l / nx)
            li = l - lj * nx

            xl = xc + (li - int(nx / 2)) * dx
            yl = yc + (lj - int(ny / 2)) * dy

            dist2 = (xl - xk) ** 2 + (yl - yk) ** 2
            cov = np.exp(-dist2 / (2 * Lx ** 2)) if Ly is None else np.exp(-dist2 / (Lx ** 2 + Ly ** 2))

            B[k, l] = cov
            B[l, k] = cov

    # variance background field
    for k in range(0, n):
        B[k, k] = varian_b

    return B
B = Bmatrix(varian_b, Lx)

# BBi = BB[int(np.floor(nx / 2)), int(np.floor(ny / 2)), :, :]
f=1
BB = np.reshape(np.asarray(B), (ny, nx, ny, nx))
pu = np.empty((ny, nx, ny, nx))
pv, uu, vv, uv = pu.copy(), pu.copy(), pu.copy(), pu.copy()
for i in range(nx):
    for j in range(ny):
        # pp = BB[j, i]
        pu[j, i] = (1 / f) * np.gradient(BB[j, i], dy)[0]
        pv[j, i] = - (1 / f) * np.gradient(BB[j, i], dx)[1]
        uu[j, i] = - (1 / f**2) * np.gradient(np.gradient(BB[j, i], dy)[0], dy)[0]
        vv[j, i] = - (1 / f**2) * np.gradient(np.gradient(BB[j, i], dx)[1], dx)[1]
        uv[j, i] = (1 / f**2) * np.gradient(np.gradient(BB[j, i], dx)[0], dy)[1]

B = np.empty((N, N))
B[0::nvar, 0::nvar] = BB.reshape((n, n))
B[1::nvar, 0::nvar] = pu.reshape((n, n))
B[0::nvar, 1::nvar] = -pu.reshape((n, n))
B[2::nvar, 0::nvar] = pv.reshape((n, n))
B[0::nvar, 2::nvar] = -pv.reshape((n, n))
B[1::nvar, 1::nvar] = uu.reshape((n, n))
B[2::nvar, 2::nvar] = vv.reshape((n, n))
B[2::nvar, 1::nvar] = uv.reshape((n, n))
B[1::nvar, 2::nvar] = uv.reshape((n, n))

coord = (int(np.floor(ny/2)), int(np.floor(nx/2)))
fig, ax = plt.subplots(nvar, nvar)
for ivar in range(nvar):
    for jvar in range(nvar):
        BB = np.reshape(B[ivar::nvar, jvar::nvar], (ny, nx, ny, nx))
        BBi = np.ma.masked_invalid(BB[coord[0], coord[1], :, :])
        conf = ax[ivar, jvar].contourf(xx, yy, BBi)
        # ax[ivar, jvar].set_xlim([-np.sqrt(2)*Lx, np.sqrt(2)*Lx])
        # ax[ivar, jvar].set_ylim([-np.sqrt(2)*Lx, np.sqrt(2)*Lx])
        plt.colorbar(conf, ax=ax[ivar, jvar])


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
    H = np.matlib.zeros((p, N))
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

            H[k, nvar * (j * nx + i)] = (1 - wx) * (1 - wy)
            H[k, nvar * (j * nx + i + 1)] = wx * (1 - wy)
            H[k, nvar * (j * nx + nx + i)] = (1 - wx) * wy
            H[k, nvar * (j * nx + nx + i + 1)] = wx * wy

            # print('Check sum: %s' % (wx * (1 - wy) + (1 - wx) * (1 - wy) + wx * wy + (1 - wx) * wy))

        else:
            raise ValueError('Observation point (%s, %s) is not within grid domain.' % (x[k], y[k]))
    return H
H = Hmatrix()

# background field vector at grid points
x_b = np.matlib.empty((N, 1))
var = (p_b, u_b, v_b)
for ivar in range(nvar):
    x_b[ivar::nvar] = np.reshape(var[ivar], (n, 1))

# background field vector
# y_b = H * x_b[0::nvar]
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

# ANALYSIS ERROR COVARIANCE MATRIX
I = np.identity(N)
A = (I - W * H) * B

# p_a = np.asarray(np.reshape(x_a, (nx, ny)))
output = ()
for ivar in range(nvar):
    output += (np.asarray(np.reshape(x_a[ivar::nvar], (ny, nx))),)


vmin, vmax = -1, 1
fig, ax = plt.subplots(2, 1)
pcol = ax[0].pcolor(xx, yy, p_b, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
ax[0].scatter(x, y, c=p_o[:p], cmap=plt.cm.seismic, vmin=vmin, vmax=vmax, edgecolors='k')
plt.colorbar(pcol, ax=ax[0])
ax[0].quiver(xx, yy, u_b, v_b)
ax[0].set_title('Background field')

pcol = ax[1].pcolor(xx, yy, output[0], cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
ax[1].scatter(x, y, c=p_o[:p], cmap=plt.cm.seismic, vmin=vmin, vmax=vmax, edgecolors='k')
plt.colorbar(pcol, ax=ax[1])
ax[1].quiver(xx, yy, output[1], output[2])
ax[1].set_title('Analysis field')

coord = (int(np.floor(ny/2)), int(np.floor(nx/2)))
fig, ax = plt.subplots(nvar, nvar)
for ivar in range(nvar):
    for jvar in range(nvar):
        AA = np.reshape(np.asarray(A)[ivar::nvar, jvar::nvar], (ny, nx, ny, nx))
        AAi = np.ma.masked_invalid(AA[coord[0], coord[1], :, :])
        conf = ax[ivar, jvar].contourf(xx, yy, AAi)
        # ax[ivar, jvar].set_xlim([-np.sqrt(2)*Lx, np.sqrt(2)*Lx])
        # ax[ivar, jvar].set_ylim([-np.sqrt(2)*Lx, np.sqrt(2)*Lx])
        plt.colorbar(conf, ax=ax[ivar, jvar])

# plt.figure()
# plt.pcolor(xx, yy, output[0], vmin=vmin, vmax=vmax)
# plt.colorbar()
#
# plt.figure()
# plt.pcolor(xx, yy, output[1], vmin=vmin, vmax=vmax)
# plt.colorbar()
#
# plt.figure()
# plt.pcolor(xx, yy, output[2], vmin=vmin, vmax=vmax)
# plt.colorbar()


C = np.matlib.ones((2,2))
C[1,0] =5
C[0,1] =2
D=np.matlib.zeros((2,2))
D[0,0] =9
D[1,1] =4
P=np.sqrt(D)*C*np.sqrt(D)