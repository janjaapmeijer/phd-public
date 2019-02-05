import sys
sys.path.insert(0,'..')
from _setup import *

import numpy as np
from scipy.interpolate import UnivariateSpline
from netCDF4 import Dataset

from OceanPy.netcdf import createNetCDF


# load ctd and gsw data
input_file = os.path.join(datadir, 'processed', 'ss9802', 'ctd', 'ss9802_ctd_gsw.nc')
output_file_gem = os.path.join(datadir, 'processed', 'ss9802', 'ctd', 'ss9802_ctd_gem.nc')
filename = os.path.split(output_file_gem)[-1]

ctd = Dataset(input_file, 'r')

dim = {
    'profile': len(ctd.dimensions['profile']),  # maximum profiles
    'plevel': len(ctd.dimensions['plevel']),  # maximum pressure levels
}

glob_attr = {
    'title': 'Sub Antarctic Front Dynamics Experiment (SAFDE) 1997-1998',
    'creator_name': 'Jan Jaap Meijer'
}

variables = {
    't':
        ('sea_water_temperature', 'f8', ('profile', 'plevel',)),
    'SP':
        ('sea_water_practical_salinity', 'f8', ('profile', 'plevel',)),
    'CT':
        ('sea_water_conservative_temperature', 'f8', ('profile', 'plevel',)),
    'pt':
        ('sea_water_potential_temperature', 'f8', ('profile', 'plevel',)),
    'SA':
        ('sea_water_absolute_salinity', 'f8', ('profile', 'plevel',)),
    'sigma0':
        ('sea_water_sigmat', 'f8', ('profile', 'plevel',)),
    'gamman':
        ('sea_water_neutral_density', 'f8', ('profile', 'plevel',))
}

try:
    nc = Dataset(output_file_gem, 'r')

except Exception:
    if sys.version_info[0] == 3:
        pass
    else:
        raise RuntimeError('Output file %s does not exist, run this script in python3 (found python%s) to create file.'
            % (filename, sys.version_info[0]))
else:

    if all(var in nc.variables for var in variables):
        raise RuntimeError('Output file %s already exists, including variables: %s' % (filename, variables))
    elif 'gamman' not in nc.variables:
        if sys.version_info[0] == 3:
            raise RuntimeError('Output file %s already exists, run this script in python2 to add variable %s.'
                  % (filename, 'gamman'))
        nc.close()
    else:
        na = [var not in nc.variables for var in variables]
        raise RuntimeError('Output file %s already exists, requested variables %s not available.' % (filename, variables[na]))

# find indices for pressure levels in CTD prodiles
pressure_levels = {}
for ip, p in enumerate(ctd['p'][:]):
    pressure_levels[p] = ip

# define reference and interest level
# p_ref = 1494
p_int = 2

# get dynamic height contours
D = np.array([ctd['deltaD'][profile, pressure_levels[p_int]] /
              ctd['g'][profile, pressure_levels[p_int]]
              for profile in range(dim['profile'])])

# Fit a spline function for the various variables at each depth level
odds = [1, 2, 9, 10, 11, 12, 27, 45, 46, 47, 75, 76, 77, 78, 101]
iodds = [istat for istat, station in enumerate(ctd['station']) if station in odds]

vars = ['t', 'SP', 'CT', 'pt', 'SA', 'sigma0'] if sys.version_info[0] == 3 else ['gamman']

splines = {}

xinterp = np.linspace(np.nanmin(D[2:]), np.nanmax(D[2:]), 100)
for p in ctd['p'][:]:
    splines[p] = {}
    for var in vars:
        mask = np.ma.masked_where((np.isnan(D)) | (np.ma.masked_invalid(ctd[var][:, pressure_levels[p]]).mask) |
                                  ([station in odds for station in ctd['station']]), D).mask
        try:
            x, y = zip(*sorted(zip(D[~mask][2:], ctd[var][:, pressure_levels[p]][~mask][2:])))
            splines[p][var] = UnivariateSpline(x, y)
        except:
            splines[p][var] = []

# create GEM
for var in vars:
    variables[var] = variables[var] + (np.ma.masked_all((dim['profile'], dim['plevel'])),)
    for ip, p in enumerate(ctd['p'][:]):
        try:
            for id, d in enumerate(D):
                variables[var][3][id, ip] = float(splines[p][var](d))
        except TypeError:
            pass


if sys.version_info[0] == 3:

    variables = {var: variables[var] for var in vars}

    # store GEM in NetCDF
    nc = createNetCDF(output_file_gem)
    nc.add_dims(dim)
    nc.add_glob_attr(glob_attr)
    nc.create_vars(variables)
    nc.close()

    print('Output file %s created, including variables: %s, run in python2 to add gamman' % (filename, list(variables.keys())))

elif sys.version_info[0] == 2:

    variables = {var: variables[var] for var in vars}

    # save data in netcdf file using OceanPy's createNetCDF class
    nc = createNetCDF(output_file_gem)
    nc.create_vars(variables)
    nc.close()

    print('Variable %s added to output file %s.' % (variables.keys(), filename))
