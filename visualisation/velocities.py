from _init import *

from OceanPy.readwrite import *
from OceanPy.projections import vincenty, haversine

from gsw import z_from_p, geostrophic_velocity, geo_strf_dyn_height

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.cm import viridis
# IMPORT CTD AND ADCP DATA
dict_ctd = read_dict(os.path.join(root, 'Analysis', 'SS9802', 'data'), 'ctd_stations.pkl', encoding='latin1')
dict_adcp = read_dict(os.path.join(root, 'Analysis', 'SS9802', 'data'), 'adcp_stations.pkl', encoding='latin1')


# TRANSECT INFORMATION
transects = {1: list(range(3, 11)), 2: list(reversed(range(11, 19))), 3: list(range(19, 28)),
             4: list(reversed(range(27, 35))), 5: list(range(37, 47)), 6: list(reversed(range(47, 58))),
             7: list(range(57, 66)), 8: list(range(69, 77)), 9: list(reversed(range(77, 85))),
             10: list(range(85, 92)), 11: list(reversed([94, 93] + list(range(95, 102))))}

# FIND INDICES OF PRESSURE LEVELS
pressure_levels = {}
for ip, p in enumerate(dict_ctd['P']):
    pressure_levels[p] = ip


# CALCULATE GEOSTROPHIC VELOCITIES
for transect in transects.keys():
    stations = np.array(transects[transect])
    nst = len(stations)

    # get coordinates from stations on transect
    lon_ctdts = np.array([dict_ctd[station]['lon'][0] for station in stations])
    lat_ctdts = np.array([dict_ctd[station]['lat'][0] for station in stations])

    # get dynamic height stream lines
    geo_strf = np.ma.masked_all((dict_ctd['P'].shape) + (nst,))
    g = geo_strf.copy()
    Vadcp = np.ma.masked_all((dict_adcp[3]['pmean'].shape) + (nst,))
    depadcp = Vadcp.copy()
    for ist, station in enumerate(transects[transect]):
        geo_strf[:, ist] = dict_ctd[station]['deltaD']
        g[:, ist] = dict_ctd[station]['g']
        Vadcp[:, ist] = dict_adcp[station]['Vmag']
        depadcp[:, ist] = dict_adcp[station]['depth']
        # geo_strf[:, ist] = geo_strf_dyn_height(dict_ctd[station]['SA'], dict_ctd[station]['CT'], dict_ctd['P'], p_ref=12)
        # deltaD[:, ist] = dict_ctd[station]['deltaD']

    # determine geostrophic velocities from ctd data
    Vg, lon_mid, lat_mid = geostrophic_velocity(geo_strf, lon_ctdts, lat_ctdts)

    # determine mid values for dynamic height
    p_int = 1000
    D = geo_strf[pressure_levels[p_int]] / g[pressure_levels[p_int]]
    i, ip1 = slice(0, -1), slice(1, None)
    D_mid = 0.5*(D[i]+D[ip1])
    D_mid = np.broadcast_to(D_mid, Vg.shape)
    D = np.broadcast_to(D, Vadcp.shape)

    # define horizontal and vertical coordinates
    nstations = np.broadcast_to(stations, Vadcp.shape)
    nstations_mid = np.broadcast_to(0.5*(stations[i]+stations[ip1]), Vg.shape)
    depth = abs(z_from_p(dict_ctd['P'][np.newaxis,].T, lat_mid[np.newaxis]))

    # plot velocity
    bounds = np.linspace(0, 0.6, 21)
    norm = BoundaryNorm(bounds, ncolors=viridis.N)
    fig, ax = plt.subplots()
    # pcol = ax.pcolor(D_mid, depth, abs(Vg))
    conf = ax.contourf(D_mid, depth, abs(Vg), 20)
    scat = ax.scatter(D, depadcp, c=Vadcp, cmap=viridis)
    ax.invert_yaxis()
    plt.title('Transect %s' %transect)
    plt.xlabel('Dynamic height')
    plt.ylabel('Depth')
    ax.set_xlim([0.26, 0.44])
    scat.set_clim([0, 0.65])
    conf.set_clim([0, 0.65])
    fig.colorbar(scat)
    # fig.savefig(os.path.join(root, 'Figures', 'SS9802', 'velocities', 'conf_geostrophic_absolute_transect%s.png' %transect), transpare
# ADCP VELOCITIES
# cmap = LinearSegmentedColormap.from_list('name', ['hotpink', 'purple', 'springgreen', 'yellow'])
# nsteps = len(transects[transect])
# fig,ax =plt.subplots()
# for ist, station in enumerate(transects[transect]):
#     ax.plot(dict_adcp[station]['Vmag'], dict_adcp[station]['depth'], label=str(station), color=cmap(ist / float(nsteps)))
# ax.invert_yaxis()
# plt.legend()


# fig, ax = plt.subplots()
# pcol = ax.pcolor(nstations_mid, depth, abs(Vg), norm=norm)
# scat = ax.scatter(nstations, depadcp, c=Vadcp, cmap=viridis, norm=norm)
# ax.invert_yaxis()
# fig.colorbar(pcol)

