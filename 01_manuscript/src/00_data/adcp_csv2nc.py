import sys
sys.path.insert(0,'..')
from _setup import *

from scipy.io import loadmat

import numpy as np
from pandas import read_csv

from datetime import datetime

from OceanPy.netcdf import createNetCDF
from netCDF4 import num2date, date2num

input_file = os.path.join(datadir, 'processed', 'ss9802', 'trawler', 'adcp', 'O&A_SS199802_adcp_ship_trawler.csv')

# READ DATA IN DATAFRAME
df = read_csv(input_file)
# project_name = df['Project'].unique()[0]
#
df = df.drop(['Survey', 'Project', 'MARLIN', 'MARLIN UUID', 'Averaging Period (sec)'], axis=1)
df.columns = ['START_TIME', 'MEAN_LAT', 'MEAN_LON', 'U_SHIP', 'V_SHIP', 'BOTTOM_DEPTH', 'DEPTH', 'U', 'V',
              'AVG_QC', 'BIN_ATTENDANCE %']
# idx = [0] + [i + 1 for i in range(len(df) - 1) if df.loc[i, 'START_TIME'] != df.loc[i + 1, 'START_TIME']]
times = df['START_TIME'].unique()
# df['START_TIME'][idx].values

nsmax = len(df.groupby(['START_TIME']).size())
npmax = max(df.groupby(['START_TIME']).size())

maskarr = np.ma.masked_all((nsmax, npmax))
p, u, v = maskarr.copy(), maskarr.copy(), maskarr.copy()
for it, (t, nobs) in enumerate(zip(times, df.groupby(['START_TIME']).size())):
    p[it, 0:nobs] = df['DEPTH'].loc[df['START_TIME'] == t].values
#     u[it, 0:nobs] = df['U'].loc[df['START_TIME'] == t].values
#     v[it, 0:nobs] = df['V'].loc[df['START_TIME'] == t].values

# READ DATA IN DICTINOARY
input_file = os.path.join(datadir, 'processed', 'ss9802', 'steve', 'ss9802_adcp_qc.mat')
output_file = os.path.join(datadir, 'processed', 'ss9802', 'adcp', 'ss9802_adcp.nc')

dict_adcp = loadmat(input_file)

# clean up data
delete = ['__globals__', '__header__', '__version__', 'ans']
for dlt in delete:
    if dlt in dict_adcp.keys():
        dict_adcp.pop(dlt)

# SET TIME STAMPS
times=[]
for t in dict_adcp['time'].T:
    day, month, year = t[:3]
    h = int(t[3])
    m = int(round(t[3] * 60) % 60)
    s = int(round(t[3] * 3600) % 60)
    times.append(datetime(int(year), int(month), int(day), h, m, s))
times = np.array(times)

np, ns = dict_adcp['press'].shape

# WRITE DATA TO NETCDF
# define dimensions of lists/ arrays
dim = {
    'profile': ns,       # maximum profiles
    'plevel': np         # maximum pressure levels
}

# define global attributes to save in NetCDF
glob_attr = {
    'title': 'Sub Antarctic Front Dynamics Experiment (SAFDE) 1997-1998'
}

# define variable datatype, dimensions and data
# try to use standard name as var_name as specified here:
# http://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html
vars = {
    'time':
        ('time', 'f8', ('profile',), times),
    'lon':
        ('longitude', 'f8', ('profile',), dict_adcp['lon'][0]),
    'lat':
        ('latitude', 'f8', ('profile',), dict_adcp['lat'][0]),
    'depth':
        ('depth', 'f8', ('profile',), dict_adcp['depth'][0]),
    'u_ship':
        ('ship eastward_sea_water_velocity', 'f8', ('profile',), dict_adcp['unav'][0]),
    'v_ship':
        ('ship northward_sea_water_velocity', 'f8', ('profile',), dict_adcp['vnav'][0]),
    'p':
        ('sea_water_pressure', 'f8', ('profile', 'plevel',), p),
    'u':
        ('eastward_sea_water_velocity', 'f8', ('profile', 'plevel',), dict_adcp['u'].T),
    'v':
        ('northward_sea_water_velocity', 'f8', ('profile', 'plevel',), dict_adcp['v'].T)
}

# save data in netcdf file using OceanPy's createNetCDF class
nc = createNetCDF(output_file)
nc.add_dims(dim)
nc.add_glob_attr(glob_attr)
nc.create_vars(vars)
nc.close()
