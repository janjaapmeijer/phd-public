
from OceanPy.maps import make_map
from OceanPy.polynomials import polyfit_2d, polyfit2d
from OceanPy.readwrite import read_dict

import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs

from scipy.io import loadmat
import numpy as np

from pandas import DataFrame

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.mlab import griddata

import cmocean
from gsw import geo_strf_dyn_height, grav, SA_from_SP, CT_from_t, pt_from_t, z_from_p, sigma0

from datetime import datetime
from astropy.time import Time

if sys.platform == 'darwin':
    root = os.path.abspath(os.path.join(os.sep, 'Users', 'Home', 'Documents', 'Jobs', 'IMAS'))

savefig = False
pathsave = os.path.join(os.path.join(root, 'Figures', 'SS9802'))
if not os.path.exists(pathsave):
    os.makedirs(pathsave)

transects = {1: list(range(2, 10)), 2: list(reversed(range(10, 18))), 3: list(range(18,27)),
             4: list(reversed(range(26, 35))), 5: list(range(36, 46)), 6: list(reversed(range(46, 57))),
             7: list(range(56, 65)), 8: list(range(68, 76)), 9: list(reversed(range(76, 84))),
             10: list(range(84, 91)), 11: list(reversed(range(92, 101)))}


# LOAD CTD DATA
# ctd = loadmat(os.path.join(root, 'Data', 'Voyages', 'SS9802', 'ctd', 'ss9802_ctd.mat'))
# met = loadmat(os.path.join(root, 'Data', 'Voyages', 'SS9802', 'underway', 'ss9802_met.mat'))
# delete = ['__globals__', '__header__', '__version__', 'ans']
# for dlt in delete:
#     if dlt in met.keys():
#         met.pop(dlt)
# write_dict(met, os.path.join(root, 'Data', 'Voyages', 'SS9802', 'met'), 'ss9802_met')
ctd = read_dict(os.path.join(root, 'Data', 'Voyages', 'SS9802', 'ctd'), 'ss9802_ctd.pkl')

#

# FIND PRESSURE LEVELS
p_ref = 1500
P = np.empty(ctd['P'].shape[0])
P[:] = np.nan
for ip, p in enumerate(ctd['P']):
    P[ip] = p
    if p == 500:
        i500 = ip
    elif p == p_ref:
        ipref = ip
print(P[i500], P[ipref])
p_ref = P[ipref]


S500db = ctd['S'][i500, 2:]
T500db = ctd['T'][i500, 2:]

pt_levels = np.linspace(1.5, 12, 22)
sig0_levels = np.linspace(26.2, 27.8, 17)

# for nts in transects.keys():
nts = 1# input('Specify the transact number, numbers 1 to 11 counted from East to West along the meander')
print(nts)

# # MAKE MAP OF STATIONS
# fig, ax, gl = make_map(projection=cimgt.OSM().crs)
# # gl.xlocator = mticker.FixedLocator([14.27, 14.28, 14.29, 14.30, 14.31, 14.32, 14.33, 14.34, 14.35, 14.36])
# # gl.ylocator = mticker.FixedLocator([55.89, 55.90, 55.91, 55.92, 55.93, 55.94, 55.95])
# extent = [137, 145, -48, -52]
# ax.set_extent(extent)
# ax.add_image(cimgt.OSM(),5)
# ax.scatter(ctd['lon'][0,2:], ctd['lat'][0,2:], transform=ccrs.PlateCarree(), zorder=2, facecolors='b')
# ax.scatter(ctd['lon'][0,:2], ctd['lat'][0,:2], transform=ccrs.PlateCarree(), zorder=3, facecolors='r')
# for istat, (xs, ys) in enumerate(zip(ctd['lon'][0], ctd['lat'][0])):
#     ax.text(xs, ys, str(ctd['profilid'][0, istat])[-3:], transform=ccrs.PlateCarree())
# ax.scatter(ctd['lon'][0, transects[nts]], ctd['lat'][0, transects[nts]], transform=ccrs.PlateCarree(), zorder=4, facecolors='yellow')
# if savefig:
#     fig.savefig(os.path.join(pathsave, 'map_transect%s.png' %nts), transparent=True)



# DYNAMIC HEIGHT and depth
nanarray = np.empty((ctd['P'].shape[0], len(transects[nts])))
nanarray[:] = np.nan
SA, CT, g, depth, pt, nprof = nanarray.copy(), nanarray.copy(), nanarray.copy(), nanarray.copy(), nanarray.copy(), nanarray.copy()
for i, profile in enumerate(transects[nts]):
    nprof[:, i] = profile
    SA[:, i] = SA_from_SP(ctd['S'][:, profile], P, ctd['lon'][0, profile], ctd['lat'][0, profile])
    CT[:, i] = CT_from_t(SA[:, i], ctd['T'][:, profile], P)
    g[:, i] = grav(ctd['lat'][0, profile], P)
    depth[:, i] = abs(z_from_p(P, ctd['lat'][0, profile]))
    pt[:, i] = pt_from_t(SA[:, i], ctd['T'][:, profile], P, p_ref)
sig0 = sigma0(SA, CT)

# for i

deltaD = geo_strf_dyn_height(SA, CT, P, p_ref=p_ref) / g
deltaD500 = np.tile(deltaD[i500], (deltaD.shape[0],1))

# g_p = grav(ctd['lat'][0, profile], P)
# SA = SA_from_SP(ctd['S'][:, profile], P, ctd['lon'][0, profile], ctd['lat'][0, profile])
# CT = CT_from_t(SA, ctd['T'][:, profile], P)
# pt = pt_from_t(SA, ctd['T'][:, profile], P, 0)
# depth = abs(z_from_p(P, ctd['lat'][0, profile]))
# deltaD = geo_strf_dyn_height(SA, CT, P)

# fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(20, 8))
# ct0 = ax[0].contourf(deltaD500, depth, CT, cmap=cmocean.cm.thermal)
# fig.colorbar(ct0, ax=ax[0])
# ax[0].set_title('Conservative Temperature')
# ax[0].set_ylabel('Depth in m')
# ax[0].invert_yaxis()
# ct1 = ax[1].contourf(deltaD500, depth, SA, cmap=cmocean.cm.haline)
# fig.colorbar(ct1, ax=ax[1])
# ax[1].set_title('Absolute Salinity')
# for i in range(len(ax)):
#     ax[i].set_xlabel(r'Dynamic height anomaly in $\frac{m^2}{s^2}$')
# plt.suptitle('Transect %s' %nts)
# if savefig:
#     fig.savefig(os.path.join(pathsave, 'CT_SA_transect%s.png' %nts), transparent=True)

fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(20, 8))
ct0 = ax[0].contourf(deltaD500, depth, pt, pt_levels,cmap=cmocean.cm.thermal)
fig.colorbar(ct0, ax=ax[0])
ax[0].set_title('Potential Temperature')
ax[0].set_ylabel('Depth in m')
ax[0].invert_yaxis()
ct = ax[1].contourf(deltaD500, depth, sig0, sig0_levels, cmap=cmocean.cm.dense)
fig.colorbar(ct, ax=ax[1])
ax[1].set_title('Potential Density')
for i in range(len(ax)):
    ax[i].set_xlabel('Profile id')
plt.suptitle('Transect %s' %nts)
if savefig:
    fig.savefig(os.path.join(pathsave, 'pt_sigma0_transect%s.png' %nts), transparent=True)

order = 3

# MAKE GRID
x = ctd['lon'][0,2:]
y = ctd['lat'][0,2:]

nx, ny = 200, 200
xi = np.linspace(x.min(), x.max(), nx)
yi = np.linspace(y.min(), y.max(), ny)
xx, yy = np.meshgrid(xi, yi)
zi = griddata(x, y, T500db, xx, yy, interp='nn')


# PLOT
Tstep = 0.2 # deg
Sstep = 0.01
Tnsteps = int((round(T500db.max()) - round(T500db.min())) / Tstep + 1)
Snsteps = int((round(S500db.max()) - round(S500db.min())) / Sstep + 1)

cmap = cm.jet
Tbounds = np.linspace(round(T500db.min()), round(T500db.max()), Tnsteps)
Sbounds = np.linspace(round(S500db.min()), round(S500db.max()), Snsteps)

Tnorm = colors.BoundaryNorm(Tbounds, cmap.N)
Snorm = colors.BoundaryNorm(Sbounds, cmap.N)


def rmse(model, observations):
    return np.sqrt(((model - observations) ** 2).mean())

fig, ax = plt.subplots(figsize=(10,10))
pc = ax.pcolor(xi, yi, zi, cmap=cmap, norm=Tnorm)
cs = ax.contour(xi, yi, zi, Tbounds, colors='k')
for ib, b in enumerate(Tbounds):
    if b in Tbounds[::5]:
        zc = cs.collections[ib]
        plt.setp(zc, linewidth=4)
ax.clabel(cs, Tbounds[0::5], inline=1, fontsize=10)
sc = ax.scatter(x, y, c=T500db, cmap=cmap, edgecolors='k')
# fig.colorbar(pc)
fig.colorbar(sc)


fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(20,8))
pcol1 = ax[0].pcolor(*polyfit2d(x, y, T500db, order=order, gridsize=(nx, ny)), cmap=cmocean.cm.thermal, norm=Tnorm)
cs1 = ax[0].contour(*polyfit2d(x, y, T500db, order=order, gridsize=(nx, ny)), Tbounds, colors='k')
pcol2 = ax[1].pcolor(*polyfit2d(x, y, S500db, order=order, gridsize=(nx, ny)), cmap=cmocean.cm.dense, norm=Snorm)
cs2 = ax[1].contour(*polyfit2d(x, y, S500db, order=order, gridsize=(nx, ny)), Sbounds, colors='k')
for ib, b in enumerate(Tbounds):
    if b in Tbounds[::5]:
        zc = cs1.collections[ib]
        plt.setp(zc, linewidth=4)
for ib, b in enumerate(Sbounds):
    if b in Sbounds[::5]:
        zc = cs2.collections[ib]
        plt.setp(zc, linewidth=4)
ax[0].clabel(cs1, Tbounds[0::5], inline=1, fontsize=10)
ax[1].clabel(cs2, Sbounds[0::5], inline=1, fontsize=10)
sc1 = ax[0].scatter(x, y, c=T500db, cmap=cmocean.cm.thermal, norm=Tnorm)
sc2 = ax[1].scatter(x, y, c=S500db, cmap=cmocean.cm.dense, norm=Snorm)
fig.colorbar(sc1, ax=ax[0])
fig.colorbar(sc2, ax=ax[1])
ax[0].set_title('T at 500 dbar')
ax[1].set_title('S at 500 dbar')

print(rmse(zi), rmse(polyfit2d(x, y, T500db, order=order, gridsize=False)[2], T500db))







# from mpl_toolkits.mplot3d import Axes3D
# fig3 = plt.figure()
# ax = fig3.gca(projection='3d')
# ax.plot_surface(*polyfit2d(x, y, T500db, order=order, gridsize=(100, 100)),
#                 cmap=cm.jet, norm=norm, rstride=1, cstride=1, alpha=0.2, linewidth=0)
# ax.set_zlim3d(T500grid.min(), T500grid.max())
# # ax.plot(x, y, T500db, 'o', label='Original data', markersize=10)
# sc = ax.scatter(x, y, T500db, c=T500db)
# plt.colorbar(sc)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# ax.set_zlabel('Z')
# ax.axis('equal')
# ax.axis('tight')