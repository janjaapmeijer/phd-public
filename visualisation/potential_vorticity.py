from _setup import *

from netCDF4 import Dataset
import numpy as np

import matplotlib.pyplot as plt

output_file = os.path.join(datadir, 'processed', 'ss9802', 'netcdf', 'ss9802_ctd_gsw.nc')
nc = Dataset(output_file, 'r')



def layer_depth(sigma0, z, intervals, ref_depth=-10, axis=-1):

    '''
    Layer depth/ thickness as function of interval.
    :param sigma0: density
    :param z: height
    :param interval: list of density intervals
    :return: layer depth and layer thickness
    '''

    if sigma0.shape != z.shape:
        raise ValueError('The shape of sigma0 and z should match, found %s and %s' % (sigma0.shape, z.shape))

    intervals = np.array(intervals)
    if intervals.ndim > 1:
        raise ValueError('The shape of interval should be one-dimensional, found %s' % intervals.ndim)
    if len(intervals) > max(sigma0.shape):
        raise ValueError('The length of interval %s should not exceed the size of sigma0[axis], found %s'
                         % (len(intervals), sigma0.shape[axis]))

    # Number of profiles
    if sigma0.ndim == 1:
        sigma0 = sigma0[np.newaxis, :]
        z = z[np.newaxis, :]

    profiles = sigma0.shape[0]

    depths = np.ones((profiles, len(intervals)))
    for profile in range(profiles):

        for iv, interval in enumerate(intervals):

            # Check if interval limit is within sigma0 limits
            if min(sigma0[profile, ]) < interval < max(sigma0[profile, ]):

                # calculate index of >10 meter reference depth
                # TODO: Ask Helen, 10m reference depth is nedessary and what about mixed layer depth
                i10, ref_dep = min(enumerate(z[profile,]), key=lambda x: abs(x[1] - ref_depth) and x[1] > ref_depth)

                # Search from the minimum error between the observations and the interval limits
                # TODO: Ask Helen, closest value or first value that passes threshold, which threshold then?
                ic, closest = min(enumerate(sigma0[profile, i10:]), key=lambda x: abs(x[1] - interval))
                ic += i10

                # Interpolate sigma0 over the range [-10 : +10] from the minimum error and
                # match the more exact sigma0 with the associated depth(z) value
                depth = np.interp(interval, sigma0[profile, ic - 1:ic + 1], z[profile, ic - 1:ic + 1])
                depths[profile, iv] = depth
            #
            else:
                depths[profile, iv] = np.nan

    return depths, abs(np.diff(depths))

sigma0 = nc['sigma0'][:]
z = -nc['z'][:]

intervals = np.linspace(26.85, 27.2, 2)


layer_dep, layer_thick = layer_depth(sigma0, z, intervals)
layer_thick = layer_thick.flatten()



# FIND INDICES OF PRESSURE LEVELS
pressure_levels = {}
for ip, p in enumerate(nc['p']):
    pressure_levels[p] = ip

i500 = pressure_levels[500]

lon, lat = nc['lon'][2:, 0], nc['lat'][2:, 0]
temp = nc['t'][2:, i500]

spacing = 0.25
xi = np.linspace(lon.min()-spacing, lon.max()+spacing, 10)
yi = np.linspace(lat.min()-spacing, lat.max()+spacing, 10)
xx, yy = np.meshgrid(xi, yi)

temp_interp = np.random.uniform(temp.min(), temp.max(), xx.shape)


plt.pcolor(xx, yy, temp_interp)
plt.scatter(lon, lat, c=temp, s=100)

# http://psc.apl.washington.edu/nonwp_projects/PHC/oi.html
# http://twister.caps.ou.edu/OBAN2016/METR5303_Lecture14.pdf
# http://www.atmosp.physics.utoronto.ca/PHY2509/ch1.pdf
# http://www.atmosp.physics.utoronto.ca/PHY2509/ch3.pdf
# http://www.atmos.millersville.edu/~lead/Obs_Data_Assimilation.html