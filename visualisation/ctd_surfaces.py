from _init import *

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import matplotlib.colors as colors

from OceanPy.maps import make_map
from OceanPy.polynomials import polyfit_2d, polyfit2d
from OceanPy.readwrite import read_dict

from scipy.io import loadmat
import numpy as np

import cmocean

def subp_figure(x, y, zz, x2=None, y2=None, z2=None, types=None, types2=None, cmaps=None, levels=None, ct_levels=None, titles=None, xlab=None, ylab=None, stitle=None, filename=None):

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
                plot = ax[r, c].scatter(x2, y2, c=z2[r, c], cmap=cmaps[r, c], norm=norm)
            if types2[r, c] == b'quiver':
                pass
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


# dict_stations = read_dict(os.path.join(root, 'Analysis', 'SS9802', 'data'), 'ss9802_stations.pkl', encoding='latin1')
# vars = ['pt', 'SA', 'lon', 'lat']
# dict_vars = {}
# for var in vars:
#     maskedarray = np.ma.masked_all((len(dict_stations.keys()) - 1,) + dict_stations[1][var].shape)
#     dict_vars[var] = maskedarray.copy()
#     for profile in list(dict_stations.keys())[:-1]:
#         dict_vars[var][profile-1,] = dict_stations[profile][var]
dict_vars = read_dict(os.path.join(root, 'Analysis', 'SS9802', 'data'), 'ctd_vars.pkl', encoding='latin1')

# FIND PRESSURE LEVELS
pressure_levels = {}
for ip, p in enumerate(dict_vars['P']):
    pressure_levels[p] = ip


# INITIALISE
order = 3

# MAKE GRID
# from matplotlib.mlab import griddata as mplgriddata
from scipy.interpolate import griddata

lon = dict_vars['lon'][2:]
lat = dict_vars['lat'][2:]

nx, ny = 200, 200
xi = np.linspace(lon.min(), lon.max(), nx)
yi = np.linspace(lat.min(), lat.max(), ny)
xx, yy = np.meshgrid(xi, yi)
# zz = np.array([[griddata(lon, lat, dict_vars['pt'][pressure_levels[400], 2:], xx, yy, interp='nn'),
#               griddata(lon, lat, dict_vars['SA'][pressure_levels[400], 2:], xx, yy, interp='nn')],
#                [griddata(lon, lat, dict_vars['pt'][pressure_levels[800], 2:], xx, yy, interp='nn'),
#               griddata(lon, lat, dict_vars['SA'][pressure_levels[800], 2:], xx, yy, interp='nn')]])
# mask = np.ma.masked_invalid(dict_trans[transect][var])
zz = np.array([[griddata((lon, lat), dict_vars['pt'][pressure_levels[400], 2:].flatten(), (xx, yy), method='linear'),
              griddata((lon, lat), dict_vars['SA'][pressure_levels[400], 2:].flatten(), (xx, yy), method='linear')],
               [griddata((lon, lat), dict_vars['pt'][pressure_levels[800], 2:].flatten(), (xx, yy), method='linear'),
              griddata((lon, lat), dict_vars['SA'][pressure_levels[800], 2:].flatten(), (xx, yy), method='linear')]])

z2 = np.array([[dict_vars['pt'][pressure_levels[400], 2:].flatten(), dict_vars['SA'][pressure_levels[400], 2:].flatten()],
              [dict_vars['pt'][pressure_levels[800], 2:].flatten(), dict_vars['SA'][pressure_levels[800], 2:].flatten()]])

cmaps = np.array([[cmocean.cm.thermal, cmocean.cm.haline], [cmocean.cm.thermal, cmocean.cm.haline]])
pt_levels = np.arange(round(z2[:,0,:].min(),1), round(z2[:,0,:].max(),1), 0.5)
SA_levels = np.arange(round(z2[:,1,:].min(),1), round(z2[:,1,:].max(),1), 0.05)
levels = np.array([[pt_levels, SA_levels], [pt_levels, SA_levels]])
titles = np.array([['Potential Temperature at %s dbar' %400, 'Absolute Salinity at %s dbar' %400],
                   ['Potential Temperature at %s dbar' %800, 'Absolute Salinity at %s dbar' %800]])
types, types[:] = np.chararray(cmaps.shape, itemsize=10), 'contourf'
types2, types2[:] = np.chararray(cmaps.shape, itemsize=10), 'scatter'
xlab = 'Longitude'
ylab = 'Latitude'
filename = os.path.join(root, 'Figures', 'SS9802', 'surfaces', 'pt_SA_400_800dbar.png')

fig, ax = subp_figure(xx, yy, zz, x2=lon, y2=lat, z2=z2, cmaps=cmaps, types=types, types2=types2, titles=titles, xlab=xlab, ylab=ylab, levels=levels, filename=None)



cmaps = np.array([[cmocean.cm.delta, cmocean.cm.dense], [cmocean.cm.delta, cmocean.cm.dense]])
titles = np.array([['Spiciness at %s dbar (ref to 0)' %400, 'Potential Density at %s dbar' %400],
                   ['Spiciness at %s dbar (ref to 0)' %800, 'Potential Density at %s dbar' %800]])

zz = np.array([[griddata((lon, lat), dict_vars['spiciness0'][pressure_levels[400], 2:].flatten(), (xx, yy), method='linear'),
                griddata((lon, lat), dict_vars['sigma0'][pressure_levels[400], 2:].flatten(), (xx, yy), method='linear')],
               [griddata((lon, lat), dict_vars['spiciness0'][pressure_levels[800], 2:].flatten(), (xx, yy), method='linear'),
                griddata((lon, lat), dict_vars['sigma0'][pressure_levels[800], 2:].flatten(), (xx, yy), method='linear')]])

z2 = np.array([[dict_vars['spiciness0'][pressure_levels[400], 2:].flatten(), dict_vars['sigma0'][pressure_levels[400], 2:].flatten()],
              [dict_vars['spiciness0'][pressure_levels[800], 2:].flatten(), dict_vars['sigma0'][pressure_levels[800], 2:].flatten()]])

spi0_levels = np.arange(round(z2[:,0,:].min(),1), round(z2[:,0,:].max(),1), 0.1)
sigma0_levels = np.arange(round(z2[:,1,:].min(),1), round(z2[:,1,:].max(),1), 0.05)
levels = np.array([[spi0_levels, sigma0_levels], [spi0_levels, sigma0_levels]])
filename = os.path.join(root, 'Figures', 'SS9802', 'surfaces', 'spi0_sigma0_400_800dbar.png')


fig, ax = subp_figure(xx, yy, zz, x2=lon, y2=lat, z2=z2, cmaps=cmaps, types=types, types2=types2, titles=titles, xlab=xlab, ylab=ylab, levels=levels, filename=None)




# # PLOT
# Tstep = 0.2 # deg
# Sstep = 0.01
# Tnsteps = int((round(T500db.max()) - round(T500db.min())) / Tstep + 1)
# Snsteps = int((round(S500db.max()) - round(S500db.min())) / Sstep + 1)
#
# cmap = cm.jet
# Tbounds = np.linspace(round(T500db.min()), round(T500db.max()), Tnsteps)
# Sbounds = np.linspace(round(S500db.min()), round(S500db.max()), Snsteps)
#
# Tnorm = colors.BoundaryNorm(Tbounds, cmap.N)
# Snorm = colors.BoundaryNorm(Sbounds, cmap.N)
#
#
# def rmse(model, observations):
#     return np.sqrt(((model - observations) ** 2).mean())
#
# fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(20,8))
# pc0 = ax[0].pcolor(xi, yi, T500grd, cmap=cmap, norm=Tnorm)
# cs0 = ax[0].contour(xi, yi, T500grd, Tbounds, colors='k')
# for ib, b in enumerate(Tbounds):
#     if b in Tbounds[::5]:
#         zc = cs0.collections[ib]
#         plt.setp(zc, linewidth=4)
# ax[0].clabel(cs0, Tbounds[0::5], inline=1, fontsize=10)
# sc0 = ax[0].scatter(x, y, c=T500db, cmap=cmap, edgecolors='k')
# fig.colorbar(sc0, ax=ax[0])
# ax[0].set_title('T at 500 dbar')
#
#
# fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(20,8))
# pcol1 = ax[0].pcolor(*polyfit2d(x, y, T500db, order=order, gridsize=(nx, ny)), cmap=cmocean.cm.thermal, norm=Tnorm)
# cs1 = ax[0].contour(*polyfit2d(x, y, T500db, order=order, gridsize=(nx, ny)), Tbounds, colors='k')
# pcol2 = ax[1].pcolor(*polyfit2d(x, y, S500db, order=order, gridsize=(nx, ny)), cmap=cmocean.cm.dense, norm=Snorm)
# cs2 = ax[1].contour(*polyfit2d(x, y, S500db, order=order, gridsize=(nx, ny)), Sbounds, colors='k')
# for ib, b in enumerate(Tbounds):
#     if b in Tbounds[::5]:
#         zc = cs1.collections[ib]
#         plt.setp(zc, linewidth=4)
# for ib, b in enumerate(Sbounds):
#     if b in Sbounds[::5]:
#         zc = cs2.collections[ib]
#         plt.setp(zc, linewidth=4)
# ax[0].clabel(cs1, Tbounds[0::5], inline=1, fontsize=10)
# ax[1].clabel(cs2, Sbounds[0::5], inline=1, fontsize=10)
# sc1 = ax[0].scatter(x, y, c=T500db, cmap=cmocean.cm.thermal, norm=Tnorm)
# sc2 = ax[1].scatter(x, y, c=S500db, cmap=cmocean.cm.dense, norm=Snorm)
# fig.colorbar(sc1, ax=ax[0])
# fig.colorbar(sc2, ax=ax[1])
# ax[0].set_title('T at 500 dbar')
# ax[1].set_title('S at 500 dbar')
#
# print(rmse(zi), rmse(polyfit2d(x, y, T500db, order=order, gridsize=False)[2], T500db))
