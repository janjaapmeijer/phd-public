from _setup import *
# import csv
from datetime import datetime
# from scipy.io import loadmat
from netCDF4 import Dataset
import numpy as np
from pandas import read_csv

# LOAD DATA
input_file = os.path.join(datadir, 'processed', 'ss9802', 'trawler', 'ctd', 'O&A_SS199802_ctd_trawler.csv')
output_file = os.path.join(datadir, 'processed', 'ss9802', 'netcdf', 'ss9802_ctd.nc')

if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))


# READ DATA IN DATAFRAME
df = read_csv(input_file)
df = df.drop(['SURVEY_NAME', 'PROJECT_NAME', 'MARLIN_ID', 'MARLIN_UUID',
              'OXYGEN_QC', 'SALINITY_QC', 'TEMPERATURE_QC'], axis=1)

# REFORMAT DATA
df['START_TIME'] = df['START_TIME'].apply(lambda t: datetime.strptime(t, '%Y-%m-%d %H:%M:%S'))
df['END_TIME'] = df['END_TIME'].apply(lambda t: datetime.strptime(t, '%Y-%m-%d %H:%M:%S'))

# MAKE LIST OF PRESSURE LEVELS
dp = 2
p_levels = np.linspace(df['PRESSURE'].min(), df['PRESSURE'].max(),
                       int((df['PRESSURE'].max() - df['PRESSURE'].min()) / dp + 1))

# (1) Store all measurements in one vector/list


# (2) Store each variable in array of stations and pressure levels

# INDEX OF FIRST OBSERVATION AT EACH STATION
idx = [0] + [i + 1 for i in range(len(df)-1) if df.loc[i, 'STATION'] != df.loc[i + 1, 'STATION']]

# DEFINE ARRAY WITH SHAPE (STATIONS, PRESSURE LEVELS)
npmax = len(p_levels)
maskarr = np.ma.masked_all((df['STATION'].max(), npmax))
temperature, salinity, oxygen = maskarr.copy(), maskarr.copy(), maskarr.copy()
for i in range(0, len(idx)-1):
    temperature[i, 0:idx[i+1]-idx[i]] = df['TEMPERATURE'].iloc[idx[i]:idx[i+1]].values
    salinity[i, 0:idx[i+1]-idx[i]] = df['SALINITY'].iloc[idx[i]:idx[i+1]].values
    oxygen[i, 0:idx[i+1]-idx[i]] = df['OXYGEN'].iloc[idx[i]:idx[i+1]].values


# (3) Store each variable in array of stations in transect, transect number and pressure levels

# TRANSECT INFORMATION
transects = {1: list(range(3, 11)), 2: list(reversed(range(11, 19))), 3: list(range(19, 28)),
             4: list(reversed(range(27, 35))), 5: list(range(37, 47)), 6: list(reversed(range(47, 58))),
             7: list(range(57, 66)), 8: list(range(69, 77)), 9: list(reversed(range(77, 85))),
             10: list(range(85, 92)), 11: list(reversed([94, 93] + list(range(95, 102))))}

ntsmax = len(transects)
nstmax = len(max(transects.items(), key=lambda x: len(x[1]))[1])

maskarr = np.ma.masked_all((ntsmax, nstmax, npmax))
temperature_ts, salinity_ts, oxygen_ts = maskarr.copy(), maskarr.copy(), maskarr.copy()
for its, transect in enumerate(transects.keys()):
    for ipf, profile in enumerate(transects[transect]):
        temperature_ts[its, ipf, ] = temperature[profile - 1]
        salinity_ts[its, ipf, ] = salinity[profile - 1]
        oxygen_ts[its, ipf, ] = oxygen[profile - 1]
        # print(its, ipf, profile - 1)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# cf = ax.contourf(test['temperature_ts'][0,].T)
# ax.invert_yaxis()
# plt.colorbar(cf)

# SET TIMES
# time_units = 'seconds since 1970-01-01 00:00:00.0 +0000'
epoch_time = datetime.utcfromtimestamp(0)
# utc_time = datetime.strptime(df['START_TIME'][0], '%d-%b-%Y %H:%M:%S')

# CREATE EMPTY NETCDF FILE
# https://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html

# profile_input = Dataset(output_file, 'w')

### PYTHON CLASS ###

dim = {
    'time': len(df['START_TIME']),  # maximum timestamps
    'profile': df['STATION'].max(), # maximum profiles
    'start_end': 2,                 # start/ end of profile
    'plevel': npmax,                # maximum pressure levels
    'transect': ntsmax,             # maximum transects
    'profile_ts': nstmax            # maximum profiles per transect
}

glob_attr = {
    'title': 'Sub Antarctic Front Dynamics Experiment (SAFDE) 1997-1998',
    'creator_name': 'Jan Jaap Meijer'
}

# try to use standard name as var_name as specified here:
# http://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html
# var_name = ['station', 'start_time', 'end_time',
#           'start_lon', 'start_lat', 'end_lon', 'end_lat', 'bottom_lon', 'bottom_lat',
#           'min_depth', 'max_depth', 'bottom_depth']

# TODO: lists longitude have different lengths
vars = {
    'station':
        ('i4', ('profile',), df['STATION'].unique()),
    'time':
        ('f8', ('profile', 'start_end', ), np.stack((df['START_TIME'].unique(), df['END_TIME'].unique()), axis=-1)),
    'longitude':
        ('f8', ('profile', 'start_end', ), np.stack((df['START_LON'].unique(), df['END_LON'].unique()), axis=-1)),
    'latitude':
        ('f8', ('profile', 'start_end', ), np.stack((df['START_LAT'].unique(), df['END_LAT'].unique()), axis=-1)),
    'sea_water_pressure':
        ('f8', ('plevel',), p_levels),
    'sea_water_temperature':
        ('f8', ('profile', 'plevel', ), temperature),
    'sea_water_practical_salinity':
        ('f8', ('profile', 'plevel', ), salinity),
    'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water':
        ('f8', ('profile', 'plevel', ), oxygen),
    'transect sea_water_temperature':
        ('f8', ('transect', 'profile_ts', 'plevel', ), temperature_ts),
    'transect sea_water_practical_salinity':
        ('f8', ('transect', 'profile_ts', 'plevel', ), salinity_ts),
    'transect mole_concentration_of_dissolved_molecular_oxygen_in_sea_water':
        ('f8', ('transect', 'profile_ts', 'plevel', ), oxygen_ts),
}

# df['START_LON'].unique() #, df['END_LON'].unique())
nc = createNetCDF(output_file)
nc.add_dims(dim)
nc.add_glob_attr(glob_attr)
nc.create_vars(vars)
nc.close()

test = Dataset(output_file, 'r')

### PYTHON CLASS ###


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


# example = Dataset(os.path.join(datadir, 'external', 'templates', 'NODC_profile_template_v1.1_2016-09-22_184951.349975.nc'), mode='r')

