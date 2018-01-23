meteo = read_dict(os.path.join(root, 'Data', 'Voyages', 'SS9802', 'met'), 'ss9802_met.pkl')

# # SET TIME STAMPS
# epoch = datetime(1900, 1, 1)        #now = datetime.datetime.utcnow()
# dt = (datetime(1970,1,1) - datetime(1900, 1, 1)).total_seconds()
# tctd = ctd['time'][0]*24*3600 - dt
#
# time_ctd = Time(tctd, format='unix', scale='utc')
# time_ctd.format = 'datetime'    # time.format = 'iso'
# # print(time.unix)      # seconds since 1970.0 (UTC)
#
# tmet = [datetime(int(meteo['time'][2, i]), int(meteo['time'][1, i]), int(meteo['time'][0, i]),
#                  int(divmod(meteo['time'][3, i], 1)[0]), int(divmod(divmod(meteo['time'][3, i], 1)[1]*60, 1)[0]),
#                  int(divmod(divmod(meteo['time'][3, i], 1)[1]*60, 1)[1]))
#         if all(np.isfinite(meteo['time'][:, i]))
#         else datetime(int(meteo['time'][2, i]), int(meteo['time'][1, i]), int(meteo['time'][0, i]))
#         for i in range(meteo['time'].shape[1])]
# time_met = Time(tmet, format='datetime', scale='utc')


# meteo = os.path.join(root, 'Data', 'Voyages', 'SS9802', 'met', 'ss9802mt.txt')

# LOAD CTD DATA
# ctd = loadmat(os.path.join(root, 'Data', 'Voyages', 'SS9802', 'ctd', 'ss9802_ctd.mat'))
# met = loadmat(os.path.join(root, 'Data', 'Voyages', 'SS9802', 'underway', 'ss9802_met.mat'))
# delete = ['__globals__', '__header__', '__version__', 'ans']
# for dlt in delete:
#     if dlt in met.keys():
#         met.pop(dlt)

# write_dict(met, os.path.join(root, 'Data', 'Voyages', 'SS9802', 'met'), 'ss9802_met')
# # def readtxt(filename):
# data = open(meteo)
# for line in data:
#     if line.startswith('GEN') and len(line.split()) == 7:
#         _, date, time, lat_deg, lat_min, lon_deg, lon_min = line.split()
#         print(date, time)
#         print(datetime.strptime((date + time).lower(), '%d-' + '%b'.lstrip('0') + '-%Y%H:%M'))
