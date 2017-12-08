from _init import *
import numpy as np
from OceanPy.readwrite import read_dict, write_dict
from scipy import interpolate
from netCDF4 import Dataset
from mayavi import mlab
from tvtk.api import tvtk
from scipy.interpolate import griddata


# 1 arc-minute resolution
bathy = Dataset(os.path.join(root, 'Data', 'Bathymetry', 'BODC', 'res1min', 'GRIDONE_2D_137.0_-52.2_145.0_-48.0.nc'))
# 30 arc-second resolution
# bathy = Dataset(os.path.join(root, 'Data', 'Bathymetry', 'BODC', 'res30sec', 'GEBCO_2014_2D_137.0_-52.2_145.0_-48.0.nc'))
# scale = 0.5
# lonx = np.linspace(min(bathy['lon']), max(bathy['lon']),
#                    int(scale * (max(bathy['lon']) - min(bathy['lon'])) / np.diff(bathy['lon']).mean()))
# laty = np.linspace(min(bathy['lat']), max(bathy['lat']),
#                    int(scale * (max(bathy['lat']) - min(bathy['lat'])) / np.diff(bathy['lat']).mean()))
# f = interpolate.interp2d(bathy['lon'], bathy['lat'], bathy['elevation'])
#
# elev_bathy = f(lonx, laty)

dict_ctd = read_dict(os.path.join(root, 'Analysis', 'SS9802', 'data'), 'ctd_stations.pkl', encoding='latin1')
lon_ctd = np.array([dict_ctd[station]['lon'][0] for station in list(dict_ctd.keys())[2:-1]])
lat_ctd = np.array([dict_ctd[station]['lat'][0] for station in list(dict_ctd.keys())[2:-1]])
dep_ctd = np.array([dict_ctd[station]['depth'] for station in list(dict_ctd.keys())[2:-1]])
sigma0 = np.array([dict_ctd[station]['sigma0'] for station in list(dict_ctd.keys())[2:-1]])

# xy = np.array(list(map(lambda x, y: [x, y], lon_ctd, lat_ctd)))

lon_flat = np.tile(lon_ctd, dep_ctd.shape[1])
lat_flat = np.tile(lat_ctd, dep_ctd.shape[1])
dep_flat, sigm_flat = np.array([]), np.array([])
for i in range(dep_ctd.shape[1]):
    dep_flat = np.append(dep_flat, dep_ctd[:, i])
    sigm_flat = np.append(sigm_flat, sigma0[:, i])

# loc = np.random.uniform(low=-500, high=-1000, size=lon_ctd.shape)
loc = np.zeros(lon_ctd.shape)

# lon_flat, lat_flat = np.array([]), np.array([])
# for lon, lat in zip(lon_ctd, lat_ctd):
#     lon_flat = np.append(lon_flat, np.ones(dep_ctd.shape[1]) * lon)
#     lat_flat = np.append(lat_flat, np.ones(dep_ctd.shape[1]) * lat)
#
# dep_flat, sigm_flat = -dep_ctd.flatten(), sigma0.flatten()
points = np.array((lon_flat, lat_flat, -dep_flat, sigm_flat)).T
# get rid of the nans
# points = points[np.where(np.isfinite(points[:,3:]))[0],]

# randomizing the order of input points
# np.random.shuffle(points)

stations = mlab.points3d(lon_ctd, lat_ctd, loc, scale_factor=0.1, color=(0,0,0))
ug = tvtk.UnstructuredGrid(points=points[:,:3])
ug.point_data.scalars = points[:, -1]
ug.point_data.scalars.name = "density"
ds = mlab.pipeline.add_dataset(ug)
mesh = mlab.pipeline.delaunay3d(ds)
iso = mlab.pipeline.iso_surface(mesh)
iso.actor.property.opacity = 0.5
iso.contour.number_of_contours = 10
iso.actor.actor.scale = (1, 1, 1e-3)
mlab.axes(xlabel='longitude', ylabel='latitude', zlabel='z')
# mlab.outline()

mlab.show()

#
#
# # https://stackoverflow.com/questions/9419451/3d-contour-plot-from-data-using-mayavi-python
# from tvtk.api import tvtk
#
# points = np.random.normal(0, 1, (1000, 3))
# ug = tvtk.UnstructuredGrid(points=points)
# ug.point_data.scalars = np.sqrt(np.sum(points**2, axis=1))
# ug.point_data.scalars.name = "value"
# ds = mlab.pipeline.add_dataset(ug)
# delaunay = mlab.pipeline.delaunay3d(ds)
# iso = mlab.pipeline.iso_surface(delaunay)
# iso.actor.property.opacity = 0.1
# iso.contour.number_of_contours = 10
#
# mlab.show()


# FIND INDICES OF PRESSURE LEVELS
pressure_levels = {}
for ip, p in enumerate(dict_ctd['P']):
    pressure_levels[p] = ip

p_ref = 1500
p_int = 1000

# get dynamic height contours
D = np.array([dict_ctd[station]['deltaD'][pressure_levels[p_int]] / dict_ctd[station]['g'][pressure_levels[p_int]]
              for station in list(dict_ctd.keys())[2:-1]])


nx, ny = 200, 200
xi = np.linspace(lon_ctd.min(), lon_ctd.max(), nx)
yi = np.linspace(lat_ctd.min(), lat_ctd.max(), ny)
xx, yy = np.meshgrid(xi, yi)
Dgrd = griddata((lon_ctd, lat_ctd), D, (xx, yy), method='linear')

# FIND DEPTH LEVELS OF DENSITY SURFACES
value = 26.9
idx = np.nanargmin(abs(sigma0-value), axis=1)
idx2 = np.nanargmin(abs(sigma0-27.15), axis=1)
dep_sig1 = np.zeros(sigma0.shape[0])
dep_sig2 = np.zeros(sigma0.shape[0])
for i in range(sigma0.shape[0]):
    dep_sig1[i] = dep_ctd[i, idx[i]]
    dep_sig2[i] = dep_ctd[i, idx2[i]]

# sig0_1 = sigma0[i, ix] for i in range




# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(loni, lati, bathy['elevation'])

loni, lati = np.meshgrid(bathy['lon'], bathy['lat'])
elev = bathy['elevation'][:]

#Mayavi requires col, row ordering. GDAL reads in row, col (i.e y, x) order
elev = np.rollaxis(elev,0,2)
loni = np.rollaxis(loni,0,2)
lati = np.rollaxis(lati,0,2)
#Mayavi has problems if the array values are not increasing
# lati = np.fliplr(lati)
# elev = np.fliplr(elev)


# read ctd data

scalefac = 1e-3


mlab.figure(bgcolor=(1,1,1), fgcolor=(0,0,0), size=(1024,786))
mlab.clf()

# plot pressure surface
# mlab.points3d(loni, lati, pres1, scale_factor=0.2)
# mlab.surf(loni, lati, pres2, warp_scale=1e-3, colormap='gist_earth', opacity=0.3)

# plot station locations
# pts = mlab.points3d(lon_ctd, lat_ctd, loc, scale_factor=0.1)
# pts.glyph.color_mode = "color_by_scalar"
# pts.glyph.glyph_source.glyph_source.center = [0,0,1000]
# pts.actor.actor.scale = (1, 1, scalefac)

# plot density points and surfaces
pts1 = mlab.points3d(lon_ctd, lat_ctd, -dep_sig1, -dep_sig1, scale_mode='none', scale_factor=0.1)
# pts2.glyph.glyph_source.glyph_source.center = [0,0,0]
pts1.actor.actor.scale = (1, 1, scalefac)

mesh = mlab.pipeline.delaunay2d(pts1)
surf = mlab.pipeline.surface(mesh, opacity=0.5)
surf.actor.actor.scale = [1, 1, scalefac]

pts2 = mlab.points3d(lon_ctd, lat_ctd, -dep_sig2, -dep_sig2, scale_mode='none', scale_factor=0.1)
pts2.actor.actor.scale = (1, 1, scalefac)
mesh = mlab.pipeline.delaunay2d(pts2)
surf2 = mlab.pipeline.surface(mesh, opacity=0.5)
surf2.actor.actor.scale = [1, 1, scalefac]

ptsD = mlab.points3d(lon_ctd, lat_ctd, loc, D, scale_factor=0.1, color=(0,0,0))
mesh = mlab.pipeline.delaunay2d(ptsD)
cont = mlab.pipeline.contour_surface(mesh, color=(0,0,0), contours=[0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42])
cont.actor.actor.scale = [1, 1, scalefac]

# plot iso surface
# http://gael-varoquaux.info/programming/mayavi-representing-an-additional-scalar-on-surfaces.html
# mesh = mlab.pipeline.delaunay2d(pts)
# surf = mlab.pipeline.surface(mesh)
# warp = mlab.pipeline.warp_scalar(mesh, warp_scale=1)
# surf = mlab.pipeline.iso_surface(warp)

# plot bathymetry
dep = mlab.surf(loni, lati, elev, colormap='viridis')#, , extent=[137, 145, -52.2, -48, -5000, 0] # 'auto' , warp_scale=-5e-4
dep.actor.actor.scale = [1, 1, scalefac]
mlab.axes(xlabel='longitude', ylabel='latitude', zlabel='z')#ranges = [137, 145, -52.2, -48, -5000, 0])
# mlab.outline(extent=[137, 145, -52.2, -48, -5000, 0])
mlab.show()



