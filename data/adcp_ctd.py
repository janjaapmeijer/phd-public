import sys
sys.path.insert(0,'..')
from _setup import *

from shutil import copyfile
from netCDF4 import Dataset, num2date
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from OceanPy.netcdf import createNetCDF

input_file_adcp = os.path.join(datadir, 'processed', 'ss9802', 'adcp', 'ss9802_adcp.nc')
input_file_ctd = os.path.join(datadir, 'processed', 'ss9802', 'netcdf', 'ss9802_ctd_gsw.nc')
output_file = os.path.join(datadir, 'processed', 'ss9802', 'adcp', 'ss9802_adcp_ctd.nc')

if os.path.isfile(input_file_adcp) and os.path.isfile(input_file_ctd):
    while True:
        try:

            # load ADCP and CTD data
            adcp = Dataset(output_file, 'r+')
            ctd = Dataset(input_file_ctd, 'r')

            # convert times to timestamps
            time_adcp = num2date(adcp['time'][:], adcp['time'].units)
            time_ctd = num2date(ctd['time'][:], ctd['time'].units)

            # average ADCP velocities during the period of the CTD cast
            nst, npl = ctd.dimensions['profile'].size, adcp.dimensions['plevel'].size
            umean, vmean = np.ma.masked_all((nst, npl)), np.ma.masked_all((nst, npl))
            pmean = np.ma.masked_all((nst, npl))
            lonmean, latmean = np.ma.masked_all((nst,)), np.ma.masked_all((nst,))
            for ist, tctd in enumerate(time_ctd):
                it = [it for it, tadcp in enumerate(time_adcp) if tctd[0] <= tadcp < tctd[1]]
                if len(it):
                    umean[ist,] = np.ma.masked_invalid(np.nanmean(adcp['u'][it], axis=0))
                    vmean[ist,] = np.ma.masked_invalid(np.nanmean(adcp['v'][it], axis=0))
                    pmean[ist,] = np.ma.masked_invalid(np.nanmean(adcp['p'][it], axis=0))
                    lonmean[ist,] = np.nanmean(adcp['lon'][it])
                    latmean[ist,] = np.nanmean(adcp['lat'][it])

            # add dimension for number of CTD profiles
            dim = {'profile_ctd': nst}  # number of CTD stations/ profiles

            vars = {
                'u_ctd':
                    ('eastward_sea_water_velocity', 'f8', ('profile_ctd', 'plevel',), umean),
                'v_ctd':
                    ('northward_sea_water_velocity', 'f8', ('profile_ctd', 'plevel',), vmean),
                'p_ctd':
                    ('sea_water_pressure', 'f8', ('profile_ctd', 'plevel',), pmean),
                'lon_ctd':
                    ('longitude', 'f8', ('profile_ctd',), lonmean),
                'lat_ctd':
                    ('latitude', 'f8', ('profile_ctd',), latmean)
            }

            # store data in netcdf
            adcp = createNetCDF(output_file)
            adcp.add_dims(dim)
            adcp.create_vars(vars)
            adcp.close()

        except FileNotFoundError:
            # duplicate input file
            copyfile(input_file_adcp, output_file)
            continue

        break

else:
    if not os.path.isfile(input_file_adcp):
        print('Input file %s does not exist, create with function adcp_csv2nc.py.' % os.path.split(input_file_adcp)[-1])
    if not os.path.isfile(input_file_ctd):
        print('Input file %s does not exist, create with function ctd_csv2nc.py.' % os.path.split(input_file_ctd)[-1])