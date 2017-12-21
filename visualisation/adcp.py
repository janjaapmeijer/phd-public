from _init import *

from scipy.io import loadmat
import numpy as np

from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from gsw import z_from_p, f, distance

from OceanPy.projections import geodetic2ecef, ecef2enu, llh2enu
from OceanPy.readwrite import read_dict, write_dict
from OceanPy.projections import haversine, vincenty
from OceanPy.find import find_in_circle, find_closest

# load adcp data and clean up
dict_adcp = loadmat(os.path.join(root, 'Data', 'Voyages', 'SS9802', 'adcp', 'ss9802_adcp_qc.mat'))

delete = ['__globals__', '__header__', '__version__', 'ans']
for dlt in delete:
    if dlt in dict_adcp.keys():
        dict_adcp.pop(dlt)

cnav = np.empty((3, dict_adcp['unav'].shape[1]), dtype='str')
for i, c in enumerate(dict_adcp['cnav']):
    cnav[i] = list(c)
dict_adcp['cnav'] = cnav

# load ctd lon, lat data
dict_ctd = read_dict(os.path.join(root, 'Analysis', 'SS9802', 'data'), 'ctd_stations.pkl', encoding='latin1')
lon_ctd = np.array([dict_ctd[station]['lon'][0] for station in list(dict_ctd.keys())[2:-1]])
lat_ctd = np.array([dict_ctd[station]['lat'][0] for station in list(dict_ctd.keys())[2:-1]])
p_ctd = dict_ctd['P']
del dict_ctd


# press_lvls = []
# idp = []
# for ip, p in enumerate(dict['press'].T):
#     press_lvls.append(np.sum(np.isfinite(p)))
#     if np.sum(np.isfinite(p)) > level:
#         idp.append(ip)
#         print(z_from_p(p, dict['lat'].T[ip]))

# uabs, vabs = dict['u'].copy(), dict['v'].copy()
# for iu, (unav, vnav) in enumerate(zip(dict['unav'].T, dict['vnav'].T)):
#     uabs[:, iu] = uabs[:, iu] + unav
#     vabs[:, iu] = vabs[:, iu] + vnav

plev_adcp = 1

# transform coordinates from latlon to x,y on f-plane
idst = slice(120, -120)
lon_adcp = dict_adcp['lon'][0, idst]
lat_adcp = dict_adcp['lat'][0, idst]
h_adcp = z_from_p(dict_adcp['press'][plev_adcp, idst], lat_adcp)

plev_ctd = find_closest(p_ctd, np.nanmean(dict_adcp['press'][plev_adcp]))
h_ctd = z_from_p(np.ones(lat_ctd.shape) * p_ctd[plev_ctd], lat_ctd)

xadcp, yadcp, zadcp = llh2enu(lon_adcp, lat_adcp, h_adcp, lon0=lon_adcp.min(), lat0=lat_adcp.min(), h0=np.nanmean(h_adcp))
xctd, yctd, zctd = llh2enu(lon_ctd, lat_ctd, h_ctd, lon0=lon_adcp.min(), lat0=lat_adcp.min(), h0=np.nanmean(h_adcp))


idxs = np.array([]).astype('int')
idxsnan = np.array([]).astype('bool')
dict_adcpvel = {}
for ist, station in enumerate(range(3, 102)):
    # TODO: ask Helen if this method is correct
    idx = find_in_circle(xadcp, yadcp, (xctd[ist], yctd[ist]), radius=5000)
    idxs = np.append(idxs, idx)

    np.nanmean(dict_adcp['u'][:, idxs], axis=1)

    dict_adcpvel[station] = {}
    umean = np.zeros(dict_adcp['press'].shape[0])
    vmean, pmean = umean.copy(), umean.copy()
    for plvl in range(dict_adcp['press'].shape[0]):
        umean[plvl] = np.nanmean(dict_adcp['u'][plvl, idst][idx])
        vmean[plvl] = np.nanmean(dict_adcp['v'][plvl, idst][idx])
        pmean[plvl] = np.nanmean(dict_adcp['press'][plvl, idst][idx])

    idxsnan = np.append(idxsnan, all(np.isnan(umean[0:24])))
    dict_adcpvel[station]['umean'], dict_adcpvel[station]['vmean'] = umean, vmean
    dict_adcpvel[station]['Vmag'] = (umean**2 + vmean**2)**(1/2)
    dict_adcpvel[station]['pmean'] = pmean
    dict_adcpvel[station]['depth'] = abs(z_from_p(pmean, lat_ctd[ist]))

# plot

fig, ax = plt.subplots(1, 2, figsize=(20,8))
ax[0].scatter(xadcp, yadcp)
ax[0].scatter(xadcp[idxs], yadcp[idxs])
ax[0].scatter(xctd, yctd, facecolors='k')
ax[0].set_xlabel('x'), ax[0].set_ylabel('y')
ax[1].scatter(lon_adcp,lat_adcp)
ax[1].scatter(lon_ctd, lat_ctd, facecolors='k')
ax[1].scatter(lon_ctd[idxsnan], lat_ctd[idxsnan], facecolors='r')
ax[1].set_xlabel('lon'), ax[1].set_ylabel('lat')
# fig.savefig(os.path.join(root, 'Figures', 'SS9802', 'map_adcp_data.png'), transparent=True)

# write adcp data to file
path = os.path.join(root, 'Analysis', 'SS9802', 'data')
filename = 'adcp_stations'
write_dict(dict_adcpvel, path, filename)

# idts = slice(0, 7)
# vincenty(lon_ctd[idts], lat_ctd[idts])

# idx = slice(57, 71)
# lon = np.asarray(list(reversed(lon_adcp[idx])))
# lat = np.asarray(list(reversed(lat_adcp[idx])))
# print(vincenty(lon, lat))
# print(haversine(lon, lat))



# plt.scatter(lon,lat)
# plt.quiver(lon, lat, dict['u'][level, 120:-120][idx], dict['v'][plev_adcp, 120:-120][idx])





nx, ny = 50, 50
xi = np.linspace(np.nanmin(x), np.nanmax(x), nx)
yi = np.linspace(np.nanmin(y), np.nanmax(y), ny)
xxi, yyi = np.meshgrid(xi, yi)

# make latlon on f-plane grid
ln = np.linspace(np.nanmin(lon), np.nanmax(lon), nx)
lt = np.linspace(np.nanmin(lat), np.nanmax(lat), ny)
lni, lti = np.meshgrid(ln, lt)

# interpolate flow speed from u,v measurements
mask = np.isfinite(dict['u'][plev_adcp, 120:-120])
uv = griddata((x[mask], y[mask]), (dict['u'][plev_adcp, 120:-120][mask]**2 + dict['v'][plev_adcp, 120:-120][mask]**2)**(1/2), (xxi, yyi), method='linear')
# from matplotlib.mlab import griddata as mplgriddata
# uv = mplgriddata(lon, lat, (dict['u'][plev_adcp, 120:-120]**2 + dict['v'][plev_adcp, 120:-120]**2)**(1/2), xxi, yyi)

# interpolate flow velocities from u,v measurements
ui = griddata((x[mask], y[mask]), dict['u'][plev_adcp, 120:-120][mask], (xxi, yyi), method='linear')
vi = griddata((x[mask], y[mask]), dict['v'][plev_adcp, 120:-120][mask], (xxi, yyi), method='linear')

# relative vorticity
dvdx = np.gradient(vi)[1] / np.gradient(xxi)[1]
dudy = np.gradient(ui)[0] / np.gradient(yyi)[0]
zeta = dvdx - dudy

# planetary/ absolute vorticity
fcor = f(lti)
eta = fcor + zeta

# continuity/ horizontally divergence free
dudx = np.gradient(ui)[0] / np.gradient(xxi)[1]
dvdy = np.gradient(vi)[1] / np.gradient(yyi)[0]
divH = dudx + dvdy


### PLOT


def subp_figure(x, y, zz, x2=None, y2=None, u2=None, v2=None, types=None, types2=None, cmaps=None, levels=None, ct_levels=None, titles=None, xlab=None, ylab=None, stitle=None, filename=None):

    from matplotlib.colors import BoundaryNorm, Normalize
    import matplotlib.cm as cm

    empty = np.empty(zz.shape[0:2], dtype='object')
    if types is None:
        types = empty.copy()
        types[:] = b'pcolor'
    if cmaps is None:
        cmaps = empty.copy()
        cmaps[:] = cm.jet
    if titles is None:
        titles = empty.copy()
    if types2 is None:
        types2 = empty.copy()

    row, col = zz.shape[0:2]
    fig, ax = plt.subplots(row, col, sharex='col', sharey='row', figsize=(20, 8))
    ax.shape = zz.shape[0:2]
    for r in range(row):
        for c in range(col):
            if types[r, c] == b'contourf':
                plot = ax[r, c].contourf(x, y, zz[r, c], cmap=cmaps[r, c]) if levels is None else \
                    ax[r, c].contourf(x, y, zz[r, c], levels[r, c], cmap=cmaps[r, c])
            if types[r, c] == b'pcolor':
                norm = Normalize(vmin=np.nanmin(zz[r, c]), vmax=np.nanmax(zz[r, c])) if levels is None else \
                    BoundaryNorm(levels[r, c], ncolors=cmaps[r, c].N)
                plot = ax[r, c].pcolor(x, y, zz[r, c], cmap=cmaps[r, c], norm=norm)
            if types2[r, c] == b'scatter':
                norm = Normalize(vmin=np.nanmin(zz[r, c]), vmax=np.nanmax(zz[r, c])) if levels is None else \
                    BoundaryNorm(levels[r, c], ncolors=cmaps[r, c].N)
                plot = ax[r, c].scatter(x2, y2, c=u2[r, c], cmap=cmaps[r, c], norm=norm)
            if types2[r, c] == b'quiver':
                qp = ax[r, c].quiver(x2, y2, u2, v2, units='inches', scale=1, pivot='mid')
                qk = ax[r, c].quiverkey(qp, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='N', coordinates='figure')
            if types2[r, c] == b'contour':
                pass
            fig.colorbar(plot, ax=ax[r, c])
            ax[r, c].set_title(titles[r, c])
            ax[row-1, c].set_xlabel(xlab)
        ax[r, c].set_xlim([x.min(), x.max()])
        ax[r, 0].set_ylabel(ylab)
        # ax[r, 0].invert_yaxis()
    plt.suptitle(stitle)
    if filename:
        fig.savefig(filename, transparent=True)
    # fig.canvas.manager.window.attributes('-topmost', 1)
    # fig.canvas.manager.window.attributes('-topmost', 0)
    return fig, ax

narrow = 3
pressure = round(np.nanmean(dict['press'][level]),1)

zz = np.array([[uv, divH], [zeta, eta]])
titles = np.array([['Current Speed', 'Horizontal Divergence'], ['Relative Vorticity', 'Absolute Vorticity']])
cmaps = np.array([[cm.jet, cm.jet], [cm.seismic, cm.seismic]])
levels = [[np.arange(0.0, 0.65, 0.05), np.arange(-2.0e-5, 2.0e-5, 0.25e-5)],
          [np.arange(-2.0e-5, 2.0e-5, 0.25e-5), np.arange(-1e-5, 1e-4, 0.25e-5)]]
types, types[:] = np.chararray(zz.shape[0:2], itemsize=10), 'contourf'
types2, types2[0,0], types2[1,0] = np.chararray(zz.shape[0:2], itemsize=10), 'quiver', 'quiver'
title = 'Velocity/ vorticity field at %s dbar' %pressure
xlab = 'Longitude'
ylab = 'Latitude'
filename = os.path.join(root, 'Figures', 'SS9802', 'surfaces', 'ADCP_velocity_vorticity_%sdbar.png' %pressure)


fig, ax = subp_figure(lni, lti, zz, types=types, titles=titles, cmaps=cmaps, stitle=title, xlab=xlab, ylab=ylab,
                      types2=types2, x2=lon, y2=lat, u2=dict['u'][plev_adcp, 120:-120], v2=dict['v'][plev_adcp, 120:-120])
ax[1, 0].quiver(lni[::narrow], lti[::narrow], ui[::narrow], vi[::narrow], color='m', units='inches', scale=1, pivot='mid')
fig.savefig(filename, transparent=True)

#
# # plot flow speed
# fig, ax = plt.subplots()
# cf = ax.contourf(lni, lti, uv, cmap=cm.jet)
# qp = ax.quiver(lon, lat, dict['u'][plev_adcp, 120:-120], dict['v'][plev_adcp, 120:-120], units='inches', scale=1, pivot='mid')
# qk = ax.quiverkey(qp, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='N', coordinates='figure')
# fig.colorbar(cf)
# plt.title('Flow speed with velocity vectors at %s dbar' %pressure)
#
# # plot relative vorticity
# fig, ax = plt.subplots()
# qp = ax.quiver(x, y, dict['u'][plev_adcp, 120:-120], dict['v'][plev_adcp, 120:-120], color='m', units='inches', scale=1, pivot='mid')
# qk = ax.quiverkey(qp, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='N', coordinates='figure')
# qp2 = ax.quiver(xxi[::narrow], yyi[::narrow], ui[::narrow], vi[::narrow], color='k', units='inches', scale=1, pivot='mid')
# # qk2 = ax.quiverkey(qp2, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='S', coordinates='figure')
# levels = np.linspace(-5e-5, 5e-5, 28)
# cf = ax.contourf(xxi, yyi, zeta, levels, cmap=cm.seismic, zorder=0)
# fig.colorbar(cf)
# plt.title('Relative vorticity with velocity vectors at %s dbar' %pressure)
#
#
# # plot absolute vorticity
# fig, ax = plt.subplots()
# levels = np.linspace(-1.3e-4, -0.9e-4, 21)
#
# cf = ax.contourf(xxi, yyi, eta, levels, cmap=cm.seismic)
# fig.colorbar(cf)
#
# # plot hozizonal divergence
# fig, ax = plt.subplots()
# levels = np.linspace(-2.0e-5, 2.0e-5, 16)
# cf = ax.contourf(xxi, yyi, divH, levels)
# fig.colorbar(cf)
