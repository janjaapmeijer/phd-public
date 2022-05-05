from _setup import *

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm, Normalize

from OceanPy.readwrite import read_dict, write_dict
from OceanPy.maps import make_map

import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs

import numpy as np

from netCDF4 import Dataset

# import cmocean
from gsw import geostrophic_velocity

# FUNCTIONS
def extremes(dict, var, ndec=None):
    import math
    key0 = list(dict_trans.keys())[0]
    minvar, maxvar = dict[key0][var][np.isfinite(dict[key0][var])].min(), dict[key0][var][np.isfinite(dict[key0][var])].max()
    for key in dict.keys():
        maxtemp = dict[key][var][np.isfinite(dict[key][var])].max()
        mintemp = dict[key][var][np.isfinite(dict[key][var])].min()
        if maxtemp > maxvar:
            maxvar = maxtemp
            if ndec:
                maxvar = math.ceil(maxvar * 10**(ndec)) / (10**ndec)
        if mintemp < minvar:
            minvar = mintemp
            if ndec:
                minvar = math.floor(minvar * 10**(ndec)) / (10**ndec)
    return minvar, maxvar
def subp_figure(x, y, zz, xyz=None, levels=None, ct_levels=None, cmaps=None, titles=None, xlab=None, ylab=None, stitle=None, type=None, filename=None):
    ''' type:   'pcol' for pcolor
                'scat' for scatter
                'cont' for contour
                'conf' for contour fill
        may have shape cmaps
    '''
    row, col = cmaps.shape
    fig, ax = plt.subplots(row, col, sharex='col', sharey='row', figsize=(20, 8))
    ax.shape = cmaps.shape
    for r in range(row):
        for c in range(col):
            if type[r, c] == b'conf':
                cs = ax[r, c].contourf(x, y, zz[r, c], cmap=cmaps[r, c]) if levels is None else \
                    ax[r, c].contourf(x, y, zz[r, c], levels[r, c], cmap=cmaps[r, c])
                if xyz is not None:
                    ct = ax[row-1, col-1].contour(xyz[0], xyz[1], xyz[2], ct_levels, colors='k')
                    ax[row-1, col-1].clabel(ct, ct_levels[0::2], fontsize=8, use_clabeltext=True)
                    for ib, b in enumerate(ct_levels):
                        zc = ct.collections[ib]
                        plt.setp(zc, linewidth=1)
            elif type[r, c] == b'scat':
                cs = ax[r, c].scatter(x[r, c], y[r, c], c=zz[r, c], cmap=cmaps, marker='+')

            else:
                norm = Normalize(vmin=np.nanmin(zz[r, c]), vmax=np.nanmax(zz[r, c])) if levels is None else \
                    BoundaryNorm(levels[r, c], ncolors=cmaps[r, c].N)
                cs = ax[r, c].pcolor(x, y, zz[r, c], cmap=cmaps[r, c], norm=norm)
                if xyz is not None:
                    ct = ax[row-1, col-1].contour(xyz[0], xyz[1], xyz[2], ct_levels, colors='k')
                    ax[row-1, col-1].clabel(ct, ct_levels[0::2], fontsize=8, use_clabeltext=True)
                    for ib, b in enumerate(ct_levels):
                        zc = ct.collections[ib]
                        plt.setp(zc, linewidth=1)
            fig.colorbar(cs, ax=ax[r, c])
            ax[r, c].set_title(titles[r, c])
            ax[row-1, c].set_xlabel(xlab)
        ax[r, c].set_xlim([x.min(), x.max()])
        ax[r, 0].set_ylabel(ylab)
        ax[r, 0].invert_yaxis()
    plt.suptitle(stitle)
    if filename:
        fig.savefig(filename, transparent=True)
    # fig.canvas.manager.window.attributes('-topmost', 1)
    # fig.canvas.manager.window.attributes('-topmost', 0)
    return fig
def figure(x, y, z, xyz=None, levels=None, ct_levels=None, cmap=None, pcol=True, title=None, xlab=None, ylab=None, flip=True, filename = None):
    fig, ax = plt.subplots(figsize=(20, 8))
    if pcol:
        norm = BoundaryNorm(levels, ncolors=cmap.N) if levels is not None else Normalize(vmin=np.nanmin(z),  vmax=np.nanmax(z))
        cf = ax.pcolor(x, y, z, cmap=cmap, norm=norm)
        if xyz is not None:
            ct = ax.contour(xyz[0], xyz[1], xyz[2], ct_levels, colors='k')
            ax.clabel(ct, ct_levels[0::2], fontsize=8, use_clabeltext=True)
            for ib, b in enumerate(ct_levels):
                zc = ct.collections[ib]
                plt.setp(zc, linewidth=1)
    else:
        cf = ax.contourf(x, y, z, levels, cmap=cmap) if levels is not None else ax.contourf(x, y, z, cmap=cmap)
        if xyz is not None:
            lines = ["-", "--", "-.", ":"]
            for i in range(int(xyz.shape[0]/3)):
                ct = ax.contour(xyz[(i*3)+0], xyz[(i*3)+1], xyz[(i*3)+2], ct_levels, colors='k', linestyles=lines[i])
                ax.clabel(ct, ct_levels[0::2], fontsize=8, use_clabeltext=True)
                for ib, b in enumerate(ct_levels):
                    zc = ct.collections[ib]
                    plt.setp(zc, linewidth=1)
    fig.colorbar(cf, ax=ax) if levels is None else fig.colorbar(cf, ax=ax, ticks=levels)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    if flip:
        ax.invert_yaxis()
    ax.set_xlabel(xlab)
    if filename:
        fig.savefig(os.path.join(pathsave, filename), transparent=True)
    # fig.canvas.manager.window.attributes('-topmost', 1)
    # fig.canvas.manager.window.attributes('-topmost', 0)
    return fig, ax

# read ctd data
# dict_stations = read_dict(os.path.join(root, 'Analysis', 'SS9802', 'data'), 'ctd_stations.pkl', encoding='latin1')
ncfile = os.path.join(datadir, 'processed', 'ss9802', 'netcdf', 'ctd.nc')
dict_stations = Dataset(ncfile, mode='r')

# TRANSECT INFORMATION
transects = {1: list(range(3, 11)), 2: list(reversed(range(11, 19))), 3: list(range(19, 28)),
             4: list(reversed(range(27, 35))), 5: list(range(37, 47)), 6: list(reversed(range(47, 58))),
             7: list(range(57, 66)), 8: list(range(69, 77)), 9: list(reversed(range(77, 85))),
             10: list(range(85, 92)), 11: list(reversed([94, 93] + list(range(95, 102))))}

# INITIALISE ARRAY
# nstmax = len(max(transects.items(), key=lambda x: len(x[1]))[1])
ntsmax = len(transects.keys())


# FIND INDICES OF PRESSURE LEVELS
pressure_levels = {}
for ip, p in enumerate(dict_stations['pressure']):
    pressure_levels[p] = ip

p_ref = 1500
p_int = 1000

print('reference level is: %f dbar\n' %dict_stations['pressure'][pressure_levels[p_ref]],
      'interest level is: %f dbar' %dict_stations['pressure'][pressure_levels[p_int]])


# GATHER PROFILES IN TRANSECTS AND DETERMINE DYNAMIC HEIGHT WITH RESPECT TO REFERENCE LEVEL
vars = ['SA', 'CT', 'pt', 'sigma0', 'spiciness0', 'depth', 'g', 'deltaD', 'gamman', 'Vg']
dict_trans = {}
for transect in transects.keys():
    # inititialise dictionary with empty array of size pressure levels and number of stations per transect
    dict_trans[transect] = {}
    maskarray = np.ma.masked_all((dict_stations['pressure'].size, len(transects[transect])))
    for var in vars[:-1]:
        dict_trans[transect][var] = maskarray.copy()
    # fill each transect array with variables from stations
    lon, lat = np.zeros(len(transects[transect])), np.zeros(len(transects[transect]))
    for ipf, profile in enumerate(transects[transect]):
        for var in vars[:-1]:
            dict_trans[transect][var][:, ipf] = dict_stations[profile][var]
        lon[ipf], lat[ipf] = dict_stations[profile]['lon'], dict_stations[profile]['lat']
    # dict_trans[transect]['Vg'] = geostrophic_velocity(dict_trans[transect]['deltaD'], lon, lat, dict_stations['P'])
    dict_trans[transect]['D'] = np.tile(dict_trans[transect]['deltaD'][pressure_levels[p_int]] / dict_trans[transect]['g'][pressure_levels[p_int]],
                                        (dict_trans[transect]['deltaD'].shape[0], 1))
    # dict_trans[transect]['P'] = np.tile(dict_stations['P'], (dict_trans[transect]['deltaD'].shape[1], 1))

    for var in vars[:-1]:
        masked = np.ma.masked_invalid(dict_trans[transect][var])
        dict_trans[transect][var]._set_mask(masked.mask)

# write_dict(dict_trans, os.path.join(root, 'Analysis', 'SS9802', 'data'), 'ss9802_transects')

# INTERPOLATE VARIABLES OVER A LINEAR SPACED DYNAMIC HEIGHT AND DEPTH/ NEUTAL DENSITY GRID
from scipy.interpolate import griddata

Dlin = np.linspace(*extremes(dict_trans, 'D'))
gammanlin = np.linspace(*extremes(dict_trans, 'gamman'), dict_stations['P'].size)
depthlin = np.linspace(*extremes(dict_trans, 'depth'), dict_stations['P'].size)

Darray, gammanarray = np.meshgrid(Dlin, gammanlin)
_, deptharray = np.meshgrid(Dlin, depthlin)

dict_transiDdep = {}
dict_transiDgamn = {}

maskedarray = np.ma.masked_all((ntsmax,) + Darray.shape)
# sigma0_3d = maskedarray.copy()
# depth_3d = maskedarray.copy()
for var in vars[:6]:
    dict_transiDgamn[var] = maskedarray.copy()
    dict_transiDdep[var] = maskedarray.copy()

    for its, transect in enumerate(transects.keys()):
        # interpolate on neutral density levels
        mask = np.ma.masked_invalid(dict_trans[transect][var])
        if var is not 'depth':
            dict_transiDgamn[var][its,] = griddata((dict_trans[transect]['D'][~mask.mask], dict_trans[transect]['gamman'][~mask.mask]),
                                                           dict_trans[transect][var][~mask.mask].flatten(), (Darray, gammanarray))

        # interpolate on isobaric levels
        dict_transiDdep[var][its,] = griddata((dict_trans[transect]['D'][~mask.mask], dict_trans[transect]['depth'][~mask.mask]),
                                             dict_trans[transect][var][~mask.mask].flatten(), (Darray, deptharray))
        dict_transiDdep[var + 'anom'] = dict_transiDdep[var].copy()
        # dict_transiDdep[var+'anom'][its] = varinterp - np.nanmean(varinterp, axis=0)
        #
        # if var is 'sigma0':
        #     sigma0_3d[transect-1] = griddata((dict_trans[transect]['D'][~mask.mask], dict_trans[transect]['depth'][~mask.mask]),
        #                                      dict_trans[transect][var][~mask.mask].flatten(), (Darray, deptharray))
        #     depth_3d[transect-1] = griddata((dict_trans[transect]['D'][~mask.mask], dict_trans[transect]['depth'][~mask.mask]),
        #                                      dict_trans[transect]['depth'][~mask.mask].flatten(), (Darray, deptharray))
    # for its in range(len(transects.keys())):
    #     dict_transiDdep[var + 'anom'][its,] = dict_transiDdep[var + 'anom'][its,] - np.nanmean(dict_transiDdep[var + 'anom'], axis=0)
    dict_transiDdep[var + 'mean'] = np.nanmean(dict_transiDdep[var], axis=0)
    for its in range(len(transects.keys())):
        dict_transiDdep[var + 'anom'][its,] = dict_transiDdep[var + 'anom'][its,] - dict_transiDdep[var + 'mean']


# DETERMINE THE VERTICAL SHIFT OF ISOPYCNALS FOR 400 AND 800 M LEVELS
def _nanargmin(arr, axis):
    try:
        return np.nanargmin(arr, axis)
    except ValueError:
        return np.nan
depths = [400, 800]
deltaz = np.zeros((len(depths), len(Dlin)))
for i, depth in enumerate(depths):
    idd = np.array(list(map(lambda x: _nanargmin(np.abs((x - depth)), axis=0), dict_transiDdep['depth'][-1,].T)))
    idd[np.isnan(idd)] = 0
    depth_11 = list(map(lambda x, ind: x[ind], dict_transiDdep['depth'][-1,].T, idd.astype('int64')))
    sigma0_11 = list(map(lambda x, ind: x[ind], dict_transiDdep['sigma0'][-1,].T, idd.astype('int64')))
    ids = np.array(list(map(lambda x, val: _nanargmin(np.abs((x - val)), axis=0), dict_transiDdep['sigma0'][0,].T, sigma0_11)))
    ids[np.isnan(ids)] = 0
    depth_1 = list(map(lambda x, ind: x[ind], dict_transiDdep['depth'][0,].T, ids.astype('int64')))

    deltaz[i] = np.array(list(map(lambda x, y: x - y, depth_11, depth_1)))


# DETERMINE CHANGE IN POTENTIAL DENSITY OVER MEANDER LENGTH
delta = np.zeros((ntsmax-1, ) + deptharray.shape)
for transect in range(len(transects.keys())-1):
    delta[transect] = dict_transiDdep['sigma0'][transect] - dict_transiDdep['sigma0'][transect+1,]


### PLOT ###
savefig = False
pathsave = os.path.join(os.path.join(root, 'Figures', 'SS9802'))
if not os.path.exists(pathsave):
    os.makedirs(pathsave)

CT_levels = np.arange(*extremes(dict_trans, 'CT', ndec=1), 0.5)
SA_levels = np.arange(*extremes(dict_trans, 'SA', ndec=1), 0.05)
pt_levels = np.arange(extremes(dict_trans, 'pt', ndec=1)[0], extremes(dict_trans, 'pt', ndec=1)[1], 0.5)
sigma0_levels = np.arange(*extremes(dict_trans, 'sigma0', ndec=1), 0.075)
spi0_levels = np.linspace(*extremes(dict_trans, 'spiciness0', ndec=1),20)
#np.arange(extremes(dict_trans, 'spiciness0', ndec=1)[0], extremes(dict_trans, 'spiciness0', ndec=1)[1], 0.075)

def xtrms(var, ndec=1):
    import math
    minvar, maxvar = math.floor(np.nanmin(var) * 10 ** (ndec)) / (10 ** ndec), \
           math.ceil(np.nanmax(var) * 10 ** (ndec)) / (10 ** ndec)
    absmax = min(abs(minvar), maxvar)
    return (-absmax, absmax)
ptanom_levels = np.linspace(*xtrms(dict_transiDdep['ptanom']), 26)
SAanom_levels = np.linspace(*xtrms(dict_transiDdep['SAanom']), 26)
spi0anom_levels = np.linspace(*xtrms(dict_transiDdep['spiciness0anom']), 26)
sigma0anom_levels = np.linspace(*xtrms(dict_transiDdep['sigma0anom']), 24)

for transect in [list(transects.keys())[0]]: # transects.keys(): #

    # # make map of stations
    # fig, ax, gl = make_map(projection=cimgt.OSM().crs)
    # # gl.xlocator = mticker.FixedLocator([14.27, 14.28, 14.29, 14.30, 14.31, 14.32, 14.33, 14.34, 14.35, 14.36])
    # # gl.ylocator = mticker.FixedLocator([55.89, 55.90, 55.91, 55.92, 55.93, 55.94, 55.95])
    # extent = [137, 145, -48, -52]
    # ax.set_extent(extent)
    # ax.add_image(cimgt.OSM(), 5)
    # for station in list(dict_stations.keys())[:-1]:
    #     ax.scatter(dict_stations[station]['lon'], dict_stations[station]['lat'], transform=ccrs.PlateCarree(), zorder=2,
    #                facecolors='b')
    #     ax.text(dict_stations[station]['lon'], dict_stations[station]['lat'], '%03d' % station,
    #             transform=ccrs.PlateCarree())
    # for station in transects[transect]:
    #     ax.scatter(dict_stations[station]['lon'], dict_stations[station]['lat'], transform=ccrs.PlateCarree(), zorder=3,
    #                facecolors='yellow')
    # if savefig:
    #     fig.savefig(os.path.join(pathsave, 'transects', 'map_transect%s.png' % transect), transparent=True)
    #
    # make plot of cons temp, abs sal, pot temp and pot density against depth
    zz = np.array([[dict_trans[transect]['pt'], dict_trans[transect]['SA']], [dict_trans[transect]['spiciness0'], dict_trans[transect]['sigma0']]])
    xyz = np.array([Darray, deptharray, dict_transiDdep['sigma0'][0,::]])
    levels = np.array([[pt_levels, SA_levels], [spi0_levels, sigma0_levels]])
    ct_levels = sigma0_levels
    cmaps = np.array([[cmocean.cm.thermal, cmocean.cm.haline], [cmocean.cm.delta, cmocean.cm.dense]])
    types = np.chararray(cmaps.shape, itemsize=4)
    types[:], types[-1,-1] = 'pcol', 'conf'
    titles = np.array([['Potential Temperature', 'Absolute Salinity'], ['Spiciness', 'Potential Density']])
    ylab = 'Depth in m'
    xlab = r'Dynamic height anomaly in $\frac{m^2}{s^2}$'
    stitle = 'Transect %s' %transect
    filename = os.path.join(pathsave, 'transects', 'pt_SA_spi0_sigma0_Ddepth_ts%s.png' %transect)

    subp_figure(dict_trans[transect]['D'], dict_trans[transect]['depth'], zz,
                xyz=xyz, levels=levels, ct_levels=ct_levels, cmaps=cmaps, titles=titles, xlab=xlab, ylab=ylab,
                stitle=stitle, type=types, filename=None)

    # make plot of anomalies in cons temp, abs sal, pot temp and pot density against dynamic height/ depth
    zz = np.array([[dict_transiDdep['pt'][transect-1] - dict_transiDdep['pt'][10],
                    dict_transiDdep['SA'][transect-1] - dict_transiDdep['SA'][10]],
                   [dict_transiDdep['spiciness0'][transect-1] - dict_transiDdep['spiciness0'][10],
                    dict_transiDdep['sigma0'][transect-1] - dict_transiDdep['sigma0'][10]]])
    # zz = np.array([[dict_transiDdep['ptmean'], dict_transiDdep['SAmean']],
    #                [dict_transiDdep['spiciness0mean'], dict_transiDdep['sigma0mean']]])
    levels = np.array([[ptanom_levels, SAanom_levels], [spi0anom_levels, sigma0anom_levels]])
    cmaps = np.array([[cm.seismic, cm.seismic], [cm.seismic, cm.seismic]])
    types = np.chararray(cmaps.shape, itemsize=4)
    types[:] = 'conf'
    titles = np.array([['Potential Temperature anomaly (DS to US)', 'Absolute Salinity anomaly (DS to US)'],
                       ['Spiciness anomaly (DS to US)', 'Potential Density anomaly (DS to US)']])
    ylab = 'Depth in m'
    xlab = r'Dynamic height anomaly in $\frac{m^2}{s^2}$'
    stitle = 'Transect %s' %transect
    filename = os.path.join(pathsave, 'transects', 'transect%s' %transect, 'pt_SA_spi0_sigma0_anomDSUS_Ddep.png')
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    subp_figure(Darray, deptharray, zz, levels=levels,cmaps=cmaps, titles=titles, xlab=xlab, ylab=ylab, stitle=stitle, type=types, filename=filename)

    zz = np.array([[dict_transiDdep['ptanom'][transect - 1], dict_transiDdep['SAanom'][transect - 1]],
                   [dict_transiDdep['spiciness0anom'][transect - 1], dict_transiDdep['sigma0anom'][transect - 1]]])
    titles = np.array([['Potential Temperature anomaly (to mean)', 'Absolute Salinity anomaly (to mean)'],
                       ['Spiciness anomaly (to mean)', 'Potential Density anomaly (to mean)']])
    filename = os.path.join(pathsave, 'transects', 'transect%s' %transect, 'pt_SA_spi0_sigma0_anommean_Ddep.png')

    subp_figure(Darray, deptharray, zz, levels=levels,cmaps=cmaps, titles=titles, xlab=xlab, ylab=ylab, stitle=stitle, type=types, filename=filename)

    # # make plot of cons temp, abs sal, pot temp and pot density against neutral density
    # zz = np.array([[dict_transiDgamn['pt'][transect-1], dict_transiDgamn['SA'][transect-1]],
    #                [dict_transiDgamn['spiciness0'][transect-1], dict_transiDgamn['sigma0'][transect-1]]])
    # levels = np.array([[pt_levels, SA_levels], [spi0_levels, sigma0_levels]])
    # cmaps = np.array([[cmocean.cm.thermal, cmocean.cm.haline], [cmocean.cm.delta, cmocean.cm.dense]])
    # types = np.chararray(cmaps.shape, itemsize=4)
    # types[:] = 'pcol'
    # titles = np.array([['Potential Temperature', 'Absolute Salinity'], ['Spiciness', 'Potential Density']])
    # ylab = 'Neutral density'
    # xlab = r'Dynamic height anomaly in $\frac{m^2}{s^2}$'
    # stitle = 'Transect %s' %transect
    # filename = os.path.join(pathsave, 'transects', 'pt_SA_spi0_sigma0_Dgamman_ts%s.png' %transect)
    #
    # subp_figure(Darray, gammanarray, zz,
    #             levels=levels, cmaps=cmaps, titles=titles, xlab=xlab, ylab=ylab, stitle=stitle, type=types, filename=None)

# make plot of change in potential density over meander length
levels = np.linspace(-0.1, 0.1, 16)
cmap = cm.seismic
xyz = np.array([Darray, deptharray, dict_transiDdep['sigma0'][-1,], Darray, deptharray, dict_transiDdep['sigma0'][0,]])

title = r'Change in Potential Density $\Delta$, summed over transects/ meander lengths'
ylab = 'Depth in m'
xlab = r'Dynamic height anomaly in $\frac{m^2}{s^2}$'
filename = os.path.join(pathsave, 'change_sigma0_over_meander_1000-1500.png')

fig, ax = figure(Darray, deptharray, delta.sum(axis=0), xyz=xyz, ct_levels=sigma0_levels, cmap=cmap, levels=levels, title=title, xlab=xlab, ylab=ylab, pcol=False)
for i, depth in enumerate(depths):
    ax.quiver(Dlin, np.ones(Dlin.shape) * depth, np.zeros(Dlin.shape), deltaz[i], scale=-1, units='y', headaxislength=3)
# ax.set_xlim([1.04, 1.29])
if savefig:
    fig.savefig(os.path.join(pathsave, filename), transparent=True)

# individual contribution of each transect to change in potential density
titles = ['Change in potential density between transects at level %s dbar' %400,
          'Change in potential density between transects at level %s dbar' %800]
filename = [os.path.join(pathsave, 'change_sigma0_at_%sdbar.png' %400),
            os.path.join(pathsave, 'change_sigma0_at_%sdbar.png' %800)]
for il, level in enumerate([pressure_levels[400], pressure_levels[800]]):
    fig, ax = plt.subplots()
    intervcm = np.linspace(0, 1, 12)
    colors = [cm.hot(x) for x in intervcm]
    for transect in range(len(transects.keys())-1):
        ax.plot(Darray[level], delta[transect][level], color=colors[transect], label='%s-%s' %(transect+1, transect+2))
    plt.legend()
    plt.grid(True)
    plt.xlabel('Dynamic Height')
    plt.ylabel(r'$\Delta\sigma_0$')
    plt.title(titles[il])
    if savefig:
        fig.savefig(os.path.join(pathsave, filename[il]), transparent=True)


# # T,S-diagram for each transect
# xlab = 'Absolute Salinity'
# ylab = 'Potential Temperature'
# row, col = (4, 3)
# fig, ax = plt.subplots(row, col, sharex='col', sharey='row', figsize=(20, 8))
# ax.shape = (row, col)
# its = 1
# for r in range(row):
#     if its <= 11:
#         for c in range(col):
#             if its <= 11:
#                 cs = ax[r, c].scatter(dict_trans[its]['SA'], dict_trans[its]['pt'],
#                                       c=dict_trans[its]['depth'], cmap='RdYlBu_r', marker='+')
#                 fig.colorbar(cs, ax=ax[r, c])
#                 ax[r, c].set_title('Transect %s' %its)
#                 ax[row - 1, c].set_xlabel(xlab)
#                 its += 1
#             else:
#                 pass
#         ax[r, 0].set_ylabel(ylab)
#         # ax[r, 0].invert_yaxis()
#         ax[r, c].set_xlim(extremes(dict_trans, 'SA'))
#         ax[r, c].set_ylim(extremes(dict_trans, 'pt'))
#     else:
#         pass
# plt.suptitle('T,S-diagram')

# plot potential temperature and absolute salinity over the length of the shiptrack
# ignore = [1, 2, 35, 36, 66, 67, 68, 92, 94, 'P']
# nprof = []
# nprof_ticks = []
# ptnprofarray = np.ma.masked_all((dict_stations['P'].shape + (1,)))
# SAnprofarray = ptnprofarray.copy()
# for profile in dict_stations.keys():
#     print(profile)
#     if profile not in ignore:
#         nprof.append(profile)
#         nprof_ticks.append('%03d' % profile)
#         ptnprofarray = np.concatenate((ptnprofarray, np.ma.masked_invalid([dict_stations[profile]['pt']]).T), axis=1)
#         SAnprofarray = np.concatenate((SAnprofarray, np.ma.masked_invalid([dict_stations[profile]['SA']]).T), axis=1)
#
# ptnprofarray, SAnprofarray = ptnprofarray[:,1:], SAnprofarray[:,1:]
#
#
# title = 'Potential Temperature for all profiles'
# ylab = 'Pressure in dbar'
# xlab = 'Profile id'
#
# fig, ax = figure(nprof, dict_stations['P'], ptnprofarray, cmap=cm.jet, xlab=xlab, ylab=ylab, title=title)
# plt.xticks(nprof, nprof_ticks, rotation='vertical')
# fig.savefig(os.path.join(pathsave, 'pt_all_profiles.png'), transparent=True)
#
# title = 'Absolute Salinity  for all profiles'
#
# fig, ax = figure(nprof, dict_stations['P'], SAnprofarray, cmap=cm.jet, xlab=xlab, ylab=ylab, title=title)
# plt.xticks(nprof, nprof_ticks, rotation='vertical')
# fig.savefig(os.path.join(pathsave, 'SA_all_profiles.png'), transparent=True)

