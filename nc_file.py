from _init import *
# import csv
from datetime import datetime
from scipy.io import loadmat
from netCDF4 import Dataset
import numpy as np

# LOAD DATA
# dict_ctd = loadmat(os.path.join(root, 'Data', 'Voyages', 'SS9802', 'ctd', 'ss9802_ctd.mat'))

input_file = os.path.join(root, 'Data', 'Voyages', 'SS9802', 'trawler', 'O&A_SS199802_ctd_trawler.csv')
output_file = os.path.join(root, 'Data', 'Voyages', 'SS9802', 'netcdf', 'ctd.nc')

from pandas import read_csv

# READ DATA IN DATAFRAME
df = read_csv(input_file)
df = df.drop(['SURVEY_NAME', 'PROJECT_NAME', 'MARLIN_ID', 'MARLIN_UUID',
              'OXYGEN_QC', 'SALINITY_QC', 'TEMPERATURE_QC'], axis=1)

# REFORMAT DATA
df['START_TIME'] = df['START_TIME'].apply(lambda t: datetime.strptime(t, '%d-%b-%Y %H:%M:%S'))
df['END_TIME'] = df['END_TIME'].apply(lambda t: datetime.strptime(t, '%d-%b-%Y %H:%M:%S'))

dp = 2
p_levels = np.linspace(df['PRESSURE'].min(), df['PRESSURE'].max(),
                       int((df['PRESSURE'].max() - df['PRESSURE'].min()) / dp + 1))

idx = [0] + [i for i in range(len(df)-1) if df.loc[i, 'STATION'] != df.loc[i + 1, 'STATION']]

maskarr = np.ma.masked_all((df['STATION'].max(), len(p_levels)))
temperature = maskarr.copy()
for i in range(len(idx)):
        temperature[i, 0:753] = df['TEMPERATURE'].iloc[idx[i]:idx[i+1]].values

# (1) Store all measurements in one vector/list

# (2) Store each variable in array of stations and pressure levels

# (3) Store each variable in array of stations in transect, transect number and pressure levels


# SET TIMES
# time_units = 'seconds since 1970-01-01 00:00:00.0 +0000'
epoch_time = datetime.utcfromtimestamp(0)
# utc_time = datetime.strptime(df['START_TIME'][0], '%d-%b-%Y %H:%M:%S')

# CREATE EMPTY NETCDF FILE
# https://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html
profile_input = Dataset(output_file, 'w')

# ADD DIMENSIONS
# (1)
profile_input.createDimension('time', len(df['START_TIME']))
# (2)
profile_input.createDimension('profile', df['STATION'].max())
profile_input.createDimension('pressure', len(p_levels))
# (3)
# profile_input.createDimension('transect', len(transects))


# ADD GLOBAL ATTRIBUTES
profile_input.Conventions = 'CF-1.6'
profile_input.Metadata_Conventions = 'Unidata Dataset Discovery v1.0'
profile_input.title = 'Sub Antarctic Front Dynamics Experiment (SAFDE) 1997-1998'

vars_1 = ['station', 'start_time', 'end_time',
          'start_lon', 'start_lat', 'end_lon', 'end_lat', 'bottom_lon', 'bottom_lat',
          'min_depth', 'max_depth', 'bottom_depth']

# ADD VARIABLES
stations = profile_input.createVariable('station', 'i4', ('profile',))
start_times = profile_input.createVariable('start_time', 'f8', ('profile',))
end_times = profile_input.createVariable('end_time', 'f8', ('profile',))
pressures = profile_input.createVariable('pressure', 'f8', ('pressure',))


# ADD VARIABLE ATTRIBUTES
stations.standard_name = 'station'
start_times.standard_name = 'start_time'
end_times.standard_name = 'end_time'
pressures.standard_name = 'pressure'

# ADD DATA TO VARIABLES
stations[:] = df['STATION'].unique()
start_times[:] = df['START_TIME'].unique()
end_times[:] = df['END_TIME'].unique()
pressures[:] = p_levels




profile_input.close()

example = Dataset(os.path.join(root, 'Data', 'NetCDF_templates', 'NODC_profile_template_v1.1_2016-09-22_184951.349975.nc'), mode='r')



test = Dataset(output_file, mode='r')