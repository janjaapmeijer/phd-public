from _setup import *

from netCDF4 import Dataset
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import interp1d

output_file = os.path.join(datadir, 'processed', 'ss9802', 'netcdf', 'ss9802_ctd_gsw.nc')
nc = Dataset(output_file, 'r')



def layer_depth(sigma0, z, interval=interval):

    # calculate index of 10 meter reference depth
    i10, ref_dep = min(enumerate(z), key=lambda x: abs(x[1] + 10))

    # # calculate index of mixed layer depth using a threshold method with de Boyer Montegut et al's criteria;
    # # a density difference of .03 kg/m^3
    # imld = i10 + next((i for i in range(len(sigma0[i10:])) if abs(sigma0[i10] - sigma0[i10+i]) > 0.03), np.nan)

    depths = []
    for iv in interval:
        if min(sigma0) < iv < max(sigma0):
            ic, closest = min(enumerate(sigma0[i10:]), key=lambda x: abs(x[1] - iv))
            ic += i10

            depth = np.interp(iv, sigma0[ic - 10:ic + 10], z[ic - 10:ic + 10])
            depths.append(depth)
            # print(iv, sigma0[ic], z[ic], depth)
        else:
            depths.append(np.nan)

    layer_depths = [abs(depths[i] - depths[i + 1]) for i in range(len(depths) - 1)]
    return layer_depths

for profile in range(len(nc.dimensions['profile'])):

    sigma0 = nc['sigma0'][profile, ]
    z = nc['z'][profile, ]
    interval = np.linspace(26.2, 27.6, 8)

    print(layer_depth(sigma0, z, interval=interval))

#

#
# z_0 = np.zeros(z.shape)
# z_mld = z_0.copy()
# z_mld[:] = z[imld]
#
# plt.plot(sigma0, z, sigma0, z_mld)


# SET TIMES
# time_units = 'seconds since 1970-01-01 00:00:00.0 +0000'
epoch_time = datetime.utcfromtimestamp(0)
# utc_time = datetime.strptime(df['START_TIME'][0], '%d-%b-%Y %H:%M:%S')

# CREATE EMPTY NETCDF FILE
# https://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html

# profile_input = Dataset(output_file, 'w')


# ADD DIMENSIONS
# (1)
profile_input.createDimension('time', len(df['START_TIME']))
# (2)
profile_input.createDimension('profile', df['STATION'].max())
profile_input.createDimension('plevel', npmax)
# (3)
profile_input.createDimension('transect', ntsmax)
profile_input.createDimension('profile_ts', nstmax)


# ADD GLOBAL ATTRIBUTES
profile_input.Conventions = 'CF-1.6'
profile_input.Metadata_Conventions = 'Unidata Dataset Discovery v1.0'
profile_input.title = 'Sub Antarctic Front Dynamics Experiment (SAFDE) 1997-1998'
profile_input.creator_name = 'Jan Jaap Meijer'
# creator_url = 'https://janjaapmeijer.github.io/AtSea/'

# CREATE VARIABLES
stations = profile_input.createVariable('station', 'i4', ('profile',))
start_times = profile_input.createVariable('starttime', 'f8', ('profile',))
end_times = profile_input.createVariable('endtime', 'f8', ('profile',))
pressures = profile_input.createVariable('pressure', 'f8', ('plevel',))
longitudes = profile_input.createVariable('longitude', 'f8', ('profile',))
latitudes = profile_input.createVariable('latitude', 'f8', ('profile',))

# bottomdepths

# (1)

# (2)
temperatures = profile_input.createVariable('temperature', 'f8', ('profile', 'plevel', ))
salinitys = profile_input.createVariable('salinity', 'f8', ('profile', 'plevel', ))
oxygens = profile_input.createVariable('oxygen', 'f8', ('profile', 'plevel', ))

# (3)
temperatures_ts = profile_input.createVariable('temperature_ts', 'f8', ('transect', 'profile_ts', 'plevel', ))
salinitys_ts = profile_input.createVariable('salinity_ts', 'f8', ('transect', 'profile_ts', 'plevel', ))
oxygens_ts = profile_input.createVariable('oxygen_ts', 'f8', ('transect', 'profile_ts', 'plevel', ))

# ADD VARIABLE ATTRIBUTES
stations.standard_name = 'station'
start_times.standard_name = 'starttime'
end_times.standard_name = 'endtime'
pressures.standard_name = 'pressure'
longitudes.standard_name = 'longitude'
latitudes.standard_name = 'latitude'

# (2)
temperatures.standard_name = 'temperature'
salinitys.standard_name = 'salinity'
oxygens.standard_name = 'oxygen'

# (3)
temperatures_ts.standard_name = 'temperature_transect'
salinitys_ts.standard_name = 'salinity_transect'
oxygens_ts.standard_name = 'oxygen_transect'

# ADD DATA TO VARIABLES
stations[:] = df['STATION'].unique()
start_times[:] = df['START_TIME'].unique()
end_times[:] = df['END_TIME'].unique()
pressures[:] = p_levels
# longitudes[:] = df['START_LON'].unique() #, df['END_LON'].unique())
# latitudes[:] = df['START_LAT'].unique() #, df['END_LAT'].unique())

# (2)
temperatures[:] = temperature
salinitys[:] = salinity
oxygens[:] = oxygen

# (3)
temperatures_ts[:] = temperature_ts
salinitys_ts[:] = salinity_ts
oxygens_ts[:] = oxygen_ts
# bottomdepths[]b

profile_input.close()


example = Dataset(os.path.join(datadir, 'external', 'templates', 'NODC_profile_template_v1.1_2016-09-22_184951.349975.nc'), mode='r')

