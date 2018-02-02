
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

