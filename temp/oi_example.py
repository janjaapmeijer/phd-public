
# def covmat():
#
#
#
# def Hmat(idim, jdim, nana, lono, lato, long, latg, bfld, obsfld):
#     '''
#
#     :param idim: x-dimension of analysis grid
#     :param jdim: y-dimension of analysis grid
#     :param nana: number of observations
#     :param lono: longitude of observations
#     :param lato: latitude of observations
#     :param long: longitude of grid points
#     :param latg: latitude of grid points
#     :param bfld: background field of grid points
#     :param obsfld: observation field
#     :return: H
#     '''

import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

datadir = os.path.abspath(os.path.join(os.sep, 'home', 'janjaapmeijer', 'Downloads'))
T500 = loadmat(os.path.join(datadir, 'optimal_interpolation', 'Ana_T500.mat'))


import numpy as np
import matplotlib.pyplot as plt
from OceanPy.interpolation import polyfit2d

# dimensions for vectors and matrices
Nx, Ny = 20, 20

NxNy = Nx * Ny #len(grid_points) * len(vars)
Nobs = 3 #len(obs)
# nns = (n, n)
# nps = (n, p)
# pns = (p, n)
# pps = (p, p)

# define grid
Lx, Ly = 1e7, 1e7
dx, dy= Lx / (Nx - 1), Ly / (Ny - 1)
dxdy = dx * dy
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
xx, yy = np.meshgrid(x, y)
# xx, yy = xx.T, yy.T

# grid center coordinates
latc, lonc = 56.5, 14.

# background field
# T_b = np.zeros((Nx, Ny))
#
#
# # define corner values.
# T_b[0, 0] = -10
# T_b[-1, 0] = -15
# T_b[0, -1] = -40
# T_b[-1, -1] = -45
f = polyfit2d([xx[0, 0], xx[-1, 0], xx[0, -1], xx[-1, -1]],
              [yy[0, 0], yy[-1, 0], yy[0, -1], yy[-1, -1]],
              [-10, -15, -40, -45], order=1)
T_b = f(xx, yy)

# # %% Fill field by bi-linear interpolation.
# for nx in range(1,Nx+1):
#     for ny in range(1,Ny+1):
#         Tnx01 = (nx-1)*T_b[-1, 0] + (Nx-nx)*T_b[0, 0]
#         TnxNy = (nx-1)*T_b[-1,-1] + (Nx-nx)*T_b[0,-1]
#         Tnxny = ((ny-1)*TnxNy + (Ny-ny)*Tnx01) / ((Nx-1)*(Ny-1))
#         T_b[nx-1, ny-1] = Tnxny


X_b = np.reshape(T_b, (NxNy, 1), order='F') + 273

T_b = np.reshape(X_b - 273, (Nx, Ny))

plt.figure()
plt.contourf(xx, yy, T_b)
plt.colorbar()

# X_b = X_b[:, np.newaxis]

# determine variance of background field and observations
# varian_b = np.var(X_b)
# varian_obs = np.var(obs)

# determine distance 1 latitude/ longitude
# lat = 56.5
# a, b = 6378e3, 6357e3
# R = np.sqrt(((a**2 * np.cos(lat * np.pi / 180))**2 +
#              (b**2 * np.sin(lat * np.pi / 180))**2) /
#             ((a * np.cos(lat * np.pi / 180))**2 +
#              (b * np.sin(lat * np.pi / 180))**2))
# clat = np.pi * R / (2 * 90)
# print(clat)
# clon = clat * np.cos(lat * np.pi / 180)

# BACKGROUND ERROR COVARIANCE MATRIX
#  Gaussian fuction to model the correlation between analysis point i and analysis point j
# r_ij is the distance between i and j
# L length scale, in the ocean mesoscale processes have a length scale on the order of the radius of deformation
# gamma_ij = np.exp(-(r_ij/L)**2)

# varb = np.var(np.mean(obs))
# B = np.matrix(np.ones(nns[0]))
# B = varb * B

Terror = 1.0
Tdelta = Lx/6
B = np.ones((NxNy, NxNy))
for ny1 in range(1, Ny+1):
    for nx1 in range(1, Nx+1):
        n1 = (ny1-1)*Nx+nx1
        for ny2 in range(1, Ny+1):
            for nx2 in range(1, Nx+1):
                n2 = (ny2 - 1) * Nx + nx2
                dx12 = x[nx1 - 1] - x[nx2 - 1]
                dy12 = y[ny1 - 1] - y[ny2 - 1]
                r12 = np.sqrt(dx12**2 + dy12**2)
                B[n1 - 1, n2 - 1] = Terror*np.exp(-r12**2 / (2 * Tdelta**2))


# if NxNy > 4:
#     BB = np.reshape(B, (Nx, Ny, Nx, Ny))
#     idx = [(np.floor(Nx/2), np.floor(Ny/2)), (0, np.floor(Ny/2)), (0, Ny-1)]
#     for i in range(len(idx)):
#         plt.figure()
#         plt.contourf(xx, yy, BB[idx[i][0], idx[i][1], :, :])
#         plt.colorbar()



nxobs = np.array([np.floor((Nx-1)/4)+1, np.floor((Nx-1)/4)+1, np.floor(3*(Nx-1)/4)+1, np.floor(Nx/2)+1])
nyobs = np.array([np.floor((Ny-1)/4)+1, np.floor(3*(Ny-1)/4)+1, np.floor(3*(Ny-1)/4)+1, np.floor(Nx/2)+1])
dxobs = np.array([0.3, 0.5, 0.7, 0.0])
dyobs = np.array([0.3, 0.5, 0.7, 0.99])
Tobs = np.array([-20, -30, -40, -50])


# %%  Convert to radiances.
Y_o  = np.zeros((Nobs,1))
xobs,yobs = np.zeros(Nobs),np.zeros(Nobs)
sigma = 4.567E-8
for nobs in range(1, Nobs+1):
    xobs[nobs-1] = (nxobs[nobs-1] - 1 + dxobs[nobs-1]) * dx
    yobs[nobs-1] = (nyobs[nobs-1] - 1 + dyobs[nobs-1]) * dy
    Y_o[nobs-1] = sigma * (Tobs[nobs-1] + 273)**4

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% Observation error covariance matrix
# %% Raderr = Teps at 220K

Teps = 1.5
Ramp = sigma * ((220 + Teps)**4 - 220**4)

R = Ramp * np.ones((Nobs, Nobs))

# %% Forward Operator or Observation Operator
#
# %% First, the linear interpolation matrix
Hinterp = np.zeros((Nobs,NxNy))
for nobs in range(1, Nobs+1):
    for ny in range(1, Ny+1):
        for nx in range(1, Nx+1):
            nn = (ny - 1) * Nx + nx
            nxyobs = (nyobs[nobs-1] - 1) * Nx + nxobs[nobs-1]
            if (nn==nxyobs): # then
                wx1 = (x[nx] - xobs[nobs-1]) * (y[ny] - yobs[nobs-1]) / dxdy
                wx2 = (xobs[nobs-1] - x[nx-1]) * (y[ny] - yobs[nobs-1]) / dxdy
                wx3 = (x[nx] - xobs[nobs-1]) * (yobs[nobs-1] - y[ny-1]) / dxdy
                wx4 = (xobs[nobs-1] - x[nx-1]) * (yobs[nobs-1] - y[ny-1]) / dxdy
                # %  fprintf('Checksum %g \n',wx1+wx2+wx3+wx4)
                Hinterp[nobs-1, nn-1] = wx1
                Hinterp[nobs-1, nn] = wx2
                Hinterp[nobs-1, nn-1+Nx] = wx3
                Hinterp[nobs-1, nn+Nx] = wx4

T_interp = np.dot(Hinterp, X_b)

H_convert = np.diag(list(map(lambda x: 4*sigma*x**3, T_interp[:, 0])))

H = np.dot(H_convert, Hinterp)

Y_b = np.reshape(list(map(lambda x: sigma*x**4, T_interp[:, 0])), (Nobs, 1))

T_diff = Tobs[0:Nobs] - (np.reshape(np.asarray(T_interp), (Nobs,)) - 273)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# %%             The Analysis
#
# %%  Analysis  Xa  = Xb  +  W * [ Y - H(Xb) ]
# %%  Gain matrix W = BH'[R+HBH']^(-1)
# %%  Analysis error covariance A = (1-WH)B
#
# %% Innovation (obs minus background field)

# if NxNy == 400:
#     print('check X_b', np.allclose(X_b, T500['X_b']))
#     print('check B', np.allclose(B, T500['B']))
#     print('check Y_o', np.allclose(Y_o, T500['Y_o']))
#     print('check R', np.allclose(R, T500['R']))
#     print('check H', np.allclose(H, T500['H']))
#     print('check Y_b', np.allclose(Y_b, T500['Y_b']))

X_b = np.matrix(X_b)
B = np.matrix(B)
Y_o = np.matrix(Y_o)
R = np.matrix(R)
H = np.matrix(H)
Y_b = np.matrix(Y_b)

d = Y_o - Y_b

# %%  Construct the gain matrix W = BH'[R+HBH']^(-1)
W = B * H.T * (R + H * B * H.T).I

# %%  Compute the analysed fields.
X_a = X_b + W * d

# %%  Analysis error covariance matrix A = (1-WH)B
I = np.identity(NxNy)
A = (I - W * H) * B

T_a = np.reshape(X_a-273, (Nx, Ny))
plt.figure()
plt.contourf(xx, yy, np.asarray(T_a))
plt.colorbar()
scat = plt.scatter(xobs, yobs, c=Tobs[:Nobs], vmin=-50, vmax=-10, s=100)
plt.colorbar(scat)
plt.title('Analysis field')

T_amlab = np.reshape(T500['X_a']-273, (T500['Nx'][0, 0], T500['Ny'][0, 0]))
plt.figure()
conf = plt.contourf(T500['xx'], T500['yy'], T_amlab)
plt.colorbar(conf)
scat = plt.scatter(T500['xobs'], T500['yobs'], c=T500['Tobs'][:-1], vmin=-50, vmax=-10, s=100)
plt.colorbar(scat)


# # information observations
# lono = np.array([0.7]) # np.array([-1, 0.5, 2])
# lato = np.array([0.85]) # np.array([1, -1, 1])
# obs = np.array([2]) # np.array([2, 8, 5])
#
# # information grid
# # if list and not array xg.flatten(), yg.flatten()
# longi = np.linspace(0, 1, 2)
# latgi = np.linspace(0, 1, 2)
# long, latg = np.meshgrid(longi, latgi)
#
#
# # if no background field, average value of all observations is used
# varg = np.ones(long.shape) * 3
#
#
#
# # BACKGROUND VECTOR
# xb = np.matrix(varg)
# # print('shape background vector', xb)
#
# # OBSERVATIONS VECTOR
# yo = np.matrix(obs)
#
#
# idim, jdim = 1, 1
# dlon, dlat = 1, 1
# x = idim / 2 + 1 + (lono - long.flatten()[0]) / dlon
# y = jdim / 2 + 1 + (lato - latg.flatten()[0]) / dlat
# i = int(x)
# j = int(y)
# if idim > i >= 1 and jdim > j >= 1:
#     wx = x - i
#     wy = y - j
#     print(wx, wy)
#
# # OBSERVATION ERROR COVARIANCE MATRIX
# varr = np.var(obs)
# R = np.matrix(np.eye(pps[0]))
# R = varr * R
#
#
#
# xm, ym, zm, Am = polyfit2d(lono, lato, obs, order=1, gridsize=(3, 3))
#
# plt.pcolor(xm, ym, zm)
# # cmap = cm.get_cmap('jet')
# scat = plt.scatter(lono, lato, c=obs, s=100)
# plt.scatter(long, latg, facecolors='w', s=100)
# plt.axis('equal')
# plt.colorbar(scat)
