from _setup import *

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy.interpolate import griddata

from netCDF4 import Dataset, num2date

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from gsw import z_from_p, f, distance
from gsw import sigma0

from OceanPy.projections import geodetic2ecef, ecef2enu, llh2enu
from OceanPy.readwrite import read_dict, write_dict
from OceanPy.projections import haversine, vincenty
from OceanPy.find import find_in_circle, find_closest

# load adcp data and clean up
# dict_adcp = loadmat(os.path.join(root, 'Data', 'Voyages', 'SS9802', 'adcp', 'ss9802_adcp_qc.mat'))
input_file_adcp = os.path.join(datadir, 'processed', 'ss9802', 'netcdf', 'ss9802_adcp.nc')
input_file_ctd = os.path.join(datadir, 'processed', 'ss9802', 'netcdf', 'ss9802_ctd_gsw.nc')

adcp = Dataset(input_file_adcp)
ctd = Dataset(input_file_ctd)

time_adcp = num2date(adcp['time'][:], adcp['time'].units)
time_ctd = num2date(ctd['time'][:], ctd['time'].units)

# calculate mean of the adcp velocities based on the time interval that the ctd was overboard
nst, npl = ctd.dimensions['profile'].size, adcp.dimensions['plevel'].size
utmean, vtmean, ptmean = np.ma.masked_all((nst, npl)), np.ma.masked_all((nst, npl)), \
                        np.ma.masked_all((nst, npl))
lonmean, latmean = np.ma.masked_all((nst,)), np.ma.masked_all((nst,))
for ist, tctd in enumerate(time_ctd):
    it = [it for it, tadcp in enumerate(time_adcp) if tctd[0] <= tadcp < tctd[1]]
    if len(it):
        utmean[ist,] = np.ma.masked_invalid(np.nanmean(adcp['u'][it], axis=0))
        vtmean[ist,] = np.ma.masked_invalid(np.nanmean(adcp['v'][it], axis=0))
        ptmean[ist,] = np.ma.masked_invalid(np.nanmean(adcp['p'][it], axis=0))
        lonmean[ist,] = np.nanmean(adcp['lon'][it])
        latmean[ist,] = np.nanmean(adcp['lat'][it])

def mixed_layer_depth(z, sig0=False, pt=False, SA=False, CT=False, smooth=False):

    # The reference depth is to avoid large part of the
    # strong diurnal cycle in the top few meters of the ocean. Dong et al. 2008
    ref = 20
    iref, ref_dep = min(enumerate(z), key=lambda x: abs(abs(x[1]) - ref))

    # smooth profiles with moving average
    if sig0 is not False:
        if smooth:
            N = 5
            sig0 = np.concatenate([np.mean(sig0[:N - 1]) * np.ones(N - 1, ),
                                   np.convolve(sig0.data, np.ones((N,)) / N, mode='valid')])
            sig0 = np.ma.masked_where(sig0 > 1e36, sig0)

        # near-surface value
        sig0_s = sig0[iref]

    if pt is not False:
        if smooth:
            N = 5
            pt = np.concatenate([np.mean(pt[:N - 1]) * np.ones(N - 1, ),
                                 np.convolve(pt.data, np.ones((N,)) / N, mode='valid')])
            pt = np.ma.masked_where(pt > 1e36, pt)

        # near-surface values
        pt_s = pt[iref]

    # Mixed Layer Depth based on de Boyer Montegut et al. 2004's property difference based criteria
    # MLD in potential density difference, fixed threshold criterion (sig0_d - sig0_s) > 0.03 kg/m^3
    if sig0 is not False and SA is False and CT is False:
        imld = iref + next((i for i in range(len(sig0[iref:]))
                            if sig0[i] - sig0_s > 0.03), np.nan)

    # MLD in potential temperature difference, fixed threshold criterion abs(pt_d - pt_s) > 0.2°C
    if pt is not False:
        imld = iref + next((i for i in range(len(pt[iref:]))
                            if abs(pt[i] - pt_s) > 0.2), np.nan)

    # MLD in potential density and potential temperature difference
    if sig0 is not False and pt is not False:
        imld = iref + next((i for i in range(len(sig0[iref:]))
                            if 0.03 < abs(sig0[i] - sig0_s) < 0.125
                            and 0.2 < abs(pt[i] - pt_s) < 1),
                           next(i for i in range(len(sig0[iref:]))
                                if sig0[i] - sig0_s > 0.03))

    # MLD in potential density with a variable threshold criterion
    if sig0 is not False and SA is not False and CT is not False:
        SA_s = SA[iref]
        CT_s = CT[iref]
        dsig0 = sigma0(SA_s, CT_s - 0.2) - sigma0(SA_s, CT_s)
        imld = iref + next((i for i in range(len(sig0[iref:]))
                            if sig0[i] - sig0_s > dsig0), np.nan)

    return imld, sig0, pt

# plot mixed layer depth
smooth = True
rows, cols = 6, 4
fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
imld = []
for ist in range(nst):
    r, c = ist // int(nst/5), int(ist/5) % 4

    sig0 = ctd['sigma0'][ist,]
    pt = ctd['pt'][ist,]
    z = ctd['z'][ist,]
    SA = ctd['SA'][ist,]
    CT = ctd['CT'][ist,]

    imld_dd, sig0_rm = mixed_layer_depth(z, sig0=sig0, smooth=smooth)[0:2]
    imld_td, pt_rm = mixed_layer_depth(z, pt=pt, smooth=smooth)[0:2]
    imld_tdd = mixed_layer_depth(z, sig0=sig0, pt=pt, smooth=smooth)[0]
    imld_vd = mixed_layer_depth(z, sig0=sig0, SA=SA, CT=CT, smooth=smooth)[0]

    ax[r,c].plot(sig0, z, 'b')
    ax[r,c].plot(sig0_rm, z, 'b--')
    # ax[r,c].plot(sig0[imld_dd], z[imld_dd], 'go')
    ax[r,c].plot(sig0[imld_tdd], z[imld_tdd], 'co')
    # ax[r,c].plot(sig0[imld_vd], z[imld_vd], 'bo')
    ax[r,c].set_ylim([-500,0])

    ax2 = ax[r,c].twiny()
    ax2.plot(pt, z, 'r')
    ax2.plot(pt[imld_td], z[imld_td], 'ro')
    ax[r,c].tick_params('x', colors='b')
    ax2.tick_params('x', colors='r')

    imld.append(imld_tdd)


# construct adcp pressure levels
padcp = np.linspace(np.nanmin(adcp['p'][:]), np.nanmax(adcp['p'][:]), adcp.dimensions['plevel'].size)

# find deepest adcp measurements
ipmax = [np.where(ptmean[ist].mask==False)[0][-1] if not all(ptmean[ist].mask) else 0
         for ist in range(nst)]
# pmax = [padcp[i] if np.isfinite(i) else np.nan for i in ipmax]

# find pressure of mixed layer depth
pmld = [ctd['p'][i] for i in imld]
imldadcp = [(np.abs(padcp - p)).argmin() for p in pmld]
# pmldadcp = [padcp[i] for i in imldadcp]

# calculate mean of velocities from surface to pressure level
utdmean = np.array([np.nanmean(utmean[ist, slice(imldadcp[ist], ipmax[ist])]) for ist in range(nst)])
vtdmean = np.array([np.nanmean(vtmean[ist, slice(imldadcp[ist], ipmax[ist])]) for ist in range(nst)])

# calculate magnitude of vectors and calculate standard deviation
vmag = np.array([np.sqrt(utmean[ist, slice(imldadcp[ist], ipmax[ist])]**2 +
                         vtmean[ist, slice(imldadcp[ist], ipmax[ist])]**2) for ist in range(nst)])

vmagstd = np.array([np.nanstd(vmag[ist]) if len(vmag[ist]) !=0 else np.nan for ist in range(nst)])

# plot velocity vectors averaged from mixed layer depth to maximum adcp depth
std_bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
colors = cm.jet(np.linspace(0, 1, len(std_bins)))
fig, ax = plt.subplots()
Q = ax.quiver(lonmean[2:], latmean[2:], utdmean[2:], vtdmean[2:], pivot='mid', units='inches')
qk = plt.quiverkey(Q, 0.85, 0.85, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
for i in range(len(std_bins)-1):
    criteria = (vmagstd[2:] > std_bins[i]) & (vmagstd[2:] <= std_bins[i + 1])
    label = '%s - %s' % (std_bins[i], std_bins[i+1])
    print(criteria, label)
    ax.scatter(lonmean[2:][criteria], latmean[2:][criteria], s=vmagstd[2:][criteria]*1e4,
               facecolors='none', label=label, color=colors[i])
ax.set_xlim([137.5, 144])
ax.set_ylim([-52.5, -48])
ax.legend(title='Standard deviation', borderpad=0.8, fontsize='medium', loc=3)

# plot
fig, ax = plt.subplots()
idi = slice(120, -120)
ax.scatter(adcp['lon'][idi], adcp['lat'][idi], facecolors='r', edgecolors='none')
ax.scatter(ctd['lon'][2:, 0], ctd['lat'][2:, 0], facecolors='none', s=100)
ax.quiver(adcp['lon'][idi], adcp['lat'][idi], adcp['u'][idi, 0], adcp['v'][idi, 0])



# xadcp, yadcp, zadcp = llh2enu(lon_adcp, lat_adcp, h_adcp, lon0=lon_adcp.min(), lat0=lat_adcp.min(), h0=np.nanmean(h_adcp))
# xctd, yctd, zctd = llh2enu(lon_ctd, lat_ctd, h_ctd, lon0=lon_adcp.min(), lat0=lat_adcp.min(), h0=np.nanmean(h_adcp))


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
