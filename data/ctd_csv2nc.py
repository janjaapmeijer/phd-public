from _setup import *

from datetime import datetime

# import csv
# from scipy.io import loadmat
import numpy as np
from pandas import read_csv, to_datetime

from netCDF4 import Dataset, date2num
from OceanPy.netcdf import createNetCDF

# LOAD DATA
input_file = os.path.join(datadir, 'processed', 'ss9802', 'trawler', 'ctd', 'O&A_SS199802_ctd_trawler.csv')
output_file = os.path.join(datadir, 'processed', 'ss9802', 'netcdf', 'ss9802_ctd.nc')

if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

if os.path.isfile(input_file):

    if os.path.isfile(output_file):
        nc = Dataset(output_file, 'r')

        print('Output file %s already exists, including variables %s.'
              % (os.path.split(output_file)[-1], ', '.join(list(nc.variables))))

    else:
        # make directories to store output file
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
        idx = [0] + [i + 1 for i in range(len(df) - 1) if df.loc[i, 'STATION'] != df.loc[i + 1, 'STATION']]

        # DEFINE ARRAY WITH SHAPE (STATIONS, PRESSURE LEVELS)
        npmax = len(p_levels)
        maskarr = np.ma.masked_all((df['STATION'].max(), npmax))
        temperature, salinity, oxygen = maskarr.copy(), maskarr.copy(), maskarr.copy()
        for i in range(0, len(idx) - 1):
            temperature[i, 0:idx[i + 1] - idx[i]] = df['TEMPERATURE'].iloc[idx[i]:idx[i + 1]].values
            salinity[i, 0:idx[i + 1] - idx[i]] = df['SALINITY'].iloc[idx[i]:idx[i + 1]].values
            oxygen[i, 0:idx[i + 1] - idx[i]] = df['OXYGEN'].iloc[idx[i]:idx[i + 1]].values

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
                temperature_ts[its, ipf,] = temperature[profile - 1]
                salinity_ts[its, ipf,] = salinity[profile - 1]
                oxygen_ts[its, ipf,] = oxygen[profile - 1]

        # WRITE DATA TO NETCDF
        # define dimensions of lists/ arrays
        dim = {
            'time': len(df['START_TIME']),  # maximum timestamps
            'profile': df['STATION'].max(),  # maximum profiles
            'start_end': 2,  # start/ end of profile
            'plevel': npmax,  # maximum pressure levels
            'transect': ntsmax,  # maximum transects
            'profile_ts': nstmax  # maximum profiles per transect
        }

        # define global attributes to save in NetCDF
        glob_attr = {
            'title': 'Sub Antarctic Front Dynamics Experiment (SAFDE) 1997-1998',
            'creator_name': 'Jan Jaap Meijer'
        }

        # define variable datatype, dimensions and data
        # try to use standard name as var_name as specified here:
        # http://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html

        # TODO: use function date2num and num2date from netCDF4 to convert datetime64 to datetime
        # https://stackoverflow.com/questions/39997314/write-times-in-netcdf-file
        # http://www.ceda.ac.uk/static/media/uploads/ncas-reading-2015/11_create_netcdf_python.pdf

        times = np.stack((to_datetime(list(df['START_TIME'][idx])).to_pydatetime(),
                          to_datetime(list(df['END_TIME'][idx])).to_pydatetime()), axis=-1)
        
        
        vars = {
            'station':
                ('station', 'i4', ('profile',), df['STATION'][idx].values),
            'time':
                ('time', 'f8', ('profile', 'start_end',), times),
            'lon':
                ('longitude', 'f8', ('profile', 'start_end',),
                 np.stack((df['START_LON'][idx], df['END_LON'][idx]), axis=-1)),
            'lat':
                ('latitude', 'f8', ('profile', 'start_end',),
                 np.stack((df['START_LAT'][idx], df['END_LAT'][idx]), axis=-1)),
            'bot_lon':
                ('bottom longitude', 'f8', ('profile',), df['BOTTOM_LON'][idx].values),
            'bot_lat':
                ('bottom latitude', 'f8', ('profile',), df['BOTTOM_LAT'][idx].values),
            'depth':
                ('depth', 'f8', ('profile',), df['BOTTOM_DEPTH'][idx].values),
            'p':
                ('sea_water_pressure', 'f8', ('plevel',), p_levels),
            't':
                ('sea_water_temperature', 'f8', ('profile', 'plevel',), temperature),
            'SP':
                ('sea_water_practical_salinity', 'f8', ('profile', 'plevel',), salinity),
            'O2':
                ('mole_concentration_of_dissolved_molecular_oxygen_in_sea_water', 'f8', ('profile', 'plevel',),
                 oxygen),
            'ts_t':
                ('transect sea_water_temperature', 'f8', ('transect', 'profile_ts', 'plevel',), temperature_ts),
            'ts_SP':
                ('transect sea_water_practical_salinity', 'f8', ('transect', 'profile_ts', 'plevel',), salinity_ts),
            'ts_O2':
                ('transect mole_concentration_of_dissolved_molecular_oxygen_in_sea_water', 'f8',
                 ('transect', 'profile_ts', 'plevel',), oxygen_ts),
        }

        # save data in netcdf file using OceanPy's createNetCDF class
        nc = createNetCDF(output_file)
        nc.add_dims(dim)
        nc.add_glob_attr(glob_attr)
        nc.create_vars(vars)
        nc.close()

        print('Output file %s created and CSV variables stored in NetCDF file as %s.'
              % (os.path.split(output_file)[-1], ', '.join(vars.keys())))

else:
    print('Input file %s does not exist, download csv file from trawler' % os.path.split(input_file)[-1])





# test = Dataset(output_file, 'r')
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# cf = ax.contourf(test['temperature_ts'][0,].T)
# ax.invert_yaxis()
# plt.colorbar(cf)