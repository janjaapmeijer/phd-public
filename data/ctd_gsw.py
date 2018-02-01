from _setup import *

from shutil import copyfile
from netCDF4 import Dataset, date2num

from OceanPy.netcdf import createNetCDF

import numpy as np

from gsw import grav, SA_from_SP, CT_from_t, pt_from_t, z_from_p, sigma0, geo_strf_dyn_height, spiciness0

input_file = os.path.join(datadir, 'processed', 'ss9802', 'netcdf', 'ss9802_ctd.nc')
output_file = os.path.join(datadir, 'processed', 'ss9802', 'netcdf', 'ss9802_ctd_gsw.nc')


#3) input exists, output exists, vars exist, gamman not
#2) input exists, output exists, vars not
#1) input exists, output not --> create output, store gsw_vars
# x 4) input not, output not, vars not


# 1) CALCULATE TEOS-10 VARIABLES IN PYTHON3
if sys.version_info[0] == 3: # python3

    gsw_vars = ('SA', 'CT', 'z', 'pt', 'sigma0', 'spiciness0', 'gamman') #, 'deltaD'

    if os.path.isfile(input_file):
        while True:
            try:
                nc = Dataset(output_file, 'r+')

                if all(var in nc.variables for var in gsw_vars):
                    print('Output file %s already exists, including GSW variables.'
                          % os.path.split(output_file)[-1])

                elif all(var in nc.variables for var in gsw_vars[:-1]):
                    print('Output file %s already exists, run this script in python2 to add variable %s.'
                          % (os.path.split(output_file)[-1], gsw_vars[-1]))
                else:

                    # change one dimension variables to two dimensions
                    p, lon, lat = nc['p'][:][np.newaxis, :], nc['lon'][:,0][:, np.newaxis], nc['lat'][:,1][:, np.newaxis]
                    
                    # convert in-situ variables to gsw variables
                    p_ref = 1500

                    SA = SA_from_SP(nc['SP'][:], p, lon, lat)
                    CT = CT_from_t(SA, nc['t'][:], p)
                    pt = pt_from_t(SA, nc['t'][:], p, p_ref)
                    # deltaD = geo_strf_dyn_height(SA, CT, p, p_ref=p_ref)
                    
                    vars = {
                        'SA':
                            ('sea_water_absolute_salinity', 'f8', ('profile', 'plevel', ), SA),
                        'CT':
                            ('sea_water_conservative_temperature', 'f8', ('profile', 'plevel', ), CT),
                        'z':
                            ('height', 'f8', ('profile', 'plevel', ), z_from_p(p, lat)),
                        'pt':
                            ('sea_water_potential_temperature', 'f8', ('profile', 'plevel', ), pt),
                        'sigma0':
                            ('sea_water_sigmat', 'f8', ('profile', 'plevel', ), sigma0(SA, CT)),
                        'spiciness0':
                            ('sea_water_spiciness', 'f8', ('profile', 'plevel', ), spiciness0(SA, CT)),
                        # 'deltaD':
                        #     ('depth', 'f8', ('profile',), deltaD),
                    }

                    # save data in netcdf file using OceanPy's createNetCDF class
                    nc = createNetCDF(output_file)
                    nc.create_vars(vars)
                    nc.close()

                    print('Output file %s created and GSW variables stored, '
                          'run this script in python2 to add variable %s.'
                          % (os.path.split(output_file)[-1], gsw_vars[-1]))

            except FileNotFoundError:
                # duplicate input file
                copyfile(input_file, output_file)
                continue

            break

    else:
        print('Input file %s does not exist, create with function ctd_csv2nc.py.' % os.path.split(input_file)[-1])





########################################################################################################################
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# cf = ax.pcolor(SA.T)
# ax.invert_yaxis()
# plt.colorbar(cf)
#
# from netCDF4 import num2date
# units = 'seconds since 1970-01-01 00:00'
# start_time = num2date(nc['time'][:,0], units=units)
# end_time = num2date(nc['time'][:,1], units=units)