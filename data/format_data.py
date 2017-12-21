from __init__ import *

import math
import numpy as np
from scipy.io import loadmat

from datetime import datetime

from netCDF4 import Dataset

# ctd_stations = Dataset(os.path.join(root, 'Data', 'NetCDF_templates', 'wodStandardLevels.nc'))
# ctd_test = Dataset(os.path.join(root, 'Data', 'NetCDF_templates', 'NODC_profile_template_v1.1_2016-09-22_184951.349975.nc'))

import pickle

def write_dict(dictionary, path, filename, protocol=pickle.HIGHEST_PROTOCOL):
    with open(os.path.join(path, filename + '.pkl'), 'wb') as f:
        pickle.dump(dictionary, f, protocol=protocol)
def read_dict(path, filename):
    with open(os.path.join(path, filename), 'rb') as f:
        return pickle.load(f)

path = os.path.join(root, 'Analysis', 'SS9802', 'data')
filename = 'ctd_stations'

# 1) CALCULATE TEOS-10 VARIABLES IN PYTHON3
if sys.version_info[0] == 3: # python3

    if os.path.exists(os.path.join(path, filename + '.pkl')):
        print('File already exists, check if gamman exists in dictionary and otherwise run this script in python2')

    else:
        from gsw import grav, SA_from_SP, CT_from_t, pt_from_t, z_from_p, sigma0, geo_strf_dyn_height, spiciness0

        # LOAD DATA
        # load ctd data and clean up
        dict_ctd = loadmat(os.path.join(root, 'Data', 'Voyages', 'SS9802', 'ctd', 'ss9802_ctd.mat'))
        delete = ['__globals__', '__header__', '__version__', 'ans']
        for dlt in delete:
            if dlt in dict_ctd.keys():
                dict_ctd.pop(dlt)

        # load adcp data and clean up
        dict_adcp = loadmat(os.path.join(root, 'Data', 'Voyages', 'SS9802', 'adcp', 'ss9802_adcp_qc.mat'))
        for dlt in delete:
            if dlt in dict_adcp.keys():
                dict_adcp.pop(dlt)

        cnav = np.empty((3, dict_adcp['unav'].shape[1]), dtype='str')
        for i, c in enumerate(dict_adcp['cnav']):
            cnav[i] = list(c)
        dict_adcp['cnav'] = cnav

        # SET TIME STAMPS
        # CTD time
        epoch = datetime(1900, 1, 1)                                        # now = datetime.datetime.utcnow()
        epoch_unix = datetime(1970, 1, 1)
        dt = (epoch_unix - epoch).total_seconds()                 # seconds between 1970 and 1990
        tctd = dict_ctd['time'][0] * 24 * 3600 - dt
        time_ctd = [datetime.utcfromtimestamp(tctd[ist]) for ist in range(len(dict_ctd['station'][0]))]

        # ADCP time
        time_adcp = []
        for t in range(dict_adcp['time'].shape[1]):
            h = math.floor(dict_adcp['time'][-1, t])
            m = math.floor((dict_adcp['time'][-1, t] - h) * 60)
            s = math.floor((((dict_adcp['time'][-1, t] - h) * 60) - m) * 60)
            if round((((dict_adcp['time'][-1, t] - h) * 60) - m) * 60) == 60:
                m += 1
                s = 0

            tadcp = np.append(list(reversed(dict_adcp['time'][:-1, t])), [h, m, s]).astype('int')
            time_adcp.append(datetime(*tuple(tadcp)))

        # find ADCP times where ship is laying still
        dict_time = {}
        shipspeed = (dict_adcp['unav'][0]**2 + dict_adcp['vnav'][0]**2)**(1/2)
        for ist, station in enumerate(dict_ctd['station'][0]):
            closest = min(time_adcp, key=lambda d: abs(d - time_ctd[ist]))
            idx = time_adcp.index(closest)
            #TODO: check with Helen if these ranges are ok?
            if shipspeed[idx - 1] <= 1.10 * shipspeed[idx]:
                idx -= 1
            idxs = [idx]
            while shipspeed[idx + 1] <= 1.10 * shipspeed[idxs[0]]:
                idx += 1
                idxs.append(idx)
            dict_time[station] = np.array(time_adcp)[idxs]

        # store original data in new dict
        vars = ['lon', 'lat', 'S', 'T', 'time']
        if filename[4:] == 'vars':
            dict_vars = {}
            for var in vars:
                if var in ['lon', 'lat']:
                    dict_vars[var] = dict_ctd[var][0,]
                elif var == 'time':
                    dict_vars[var] = time_ctd
                else:
                    dict_vars[var] = dict_ctd[var]
        elif filename[4:] == 'stations':
            dict_stations = {}
            for istat, station in enumerate(dict_ctd['station'][0, :]):
                dict_stations[station] = {}
                for var in vars:
                    if var is 'time':
                        # TODO: datetime object not able to read/write with pickle in Python 2
                        dict_stations[station][var] = time_ctd[istat]
                        dict_stations[station][var + '_adcp'] = dict_time[station]
                    else:
                        dict_stations[station][var] = dict_ctd[var][:, istat]

        # format pressure levels and store in dict
        P = np.zeros(dict_ctd['P'].shape[0])
        P[:] = np.nan
        for ip, p in enumerate(dict_ctd['P']):
            P[ip] = p
        if filename[4:] == 'vars':
            dict_vars['P'] = P
            Parray = np.tile(dict_vars['P'], (dict_vars['S'].shape[1], 1)).T
        elif filename[4:] == 'stations':
            dict_stations['P'] = P

        # calculate TEOS-10 variables and store in dict
        p_ref = 1500
        if filename[4:] == 'vars':
            dict_vars['SA'] = SA_from_SP(dict_vars['S'], Parray, dict_vars['lon'], dict_vars['lat'])
            dict_vars['CT'] = CT_from_t(dict_vars['SA'], dict_vars['T'], Parray)
            dict_vars['depth'] = abs(z_from_p(Parray, dict_vars['lat']))
            dict_vars['pt'] = pt_from_t(dict_vars['SA'], dict_vars['T'], Parray, p_ref)
            dict_vars['sigma0'] = sigma0(dict_vars['SA'], dict_vars['CT'])
            dict_vars['spiciness0'] = spiciness0(dict_vars['SA'], dict_vars['CT'])

            # save dictionary to file
            write_dict(dict_vars, path, filename)
            print('data stored in file')

        elif filename[4:] == 'stations':
            for station in dict_ctd['station'][0, :]:
                dict_stations[station]['SA'] = SA_from_SP(dict_stations[station]['S'], dict_stations['P'],
                                                          dict_stations[station]['lon'], dict_stations[station]['lat'])
                dict_stations[station]['CT'] = CT_from_t(dict_stations[station]['SA'], dict_stations[station]['T'],
                                                         dict_stations['P'])
                dict_stations[station]['g'] = grav(dict_stations[station]['lat'], dict_stations['P'])
                dict_stations[station]['depth'] = abs(z_from_p(dict_stations['P'], dict_stations[station]['lat']))
                dict_stations[station]['pt'] = pt_from_t(dict_stations[station]['SA'], dict_stations[station]['T'],
                                                         dict_stations['P'], p_ref)
                dict_stations[station]['sigma0'] = sigma0(dict_stations[station]['SA'], dict_stations[station]['CT'])
                dict_stations[station]['spiciness0'] = spiciness0(dict_stations[station]['SA'], dict_stations[station]['CT'])
                dict_stations[station]['deltaD'] = geo_strf_dyn_height(dict_stations[station]['SA'], dict_stations[station]['CT'],
                                                                       dict_stations['P'], p_ref=p_ref)

            # save dictionary to file
            write_dict(dict_stations, path, filename, protocol=2) # protocol 2 allows python2
            print('data stored in file')

# 2) CALCULATE NEUTRAL DENSITY IN PYTHON2
if sys.version_info[0] == 2:  # python2

    from pygamman import gamman as nds

    if os.path.exists(os.path.join(path, filename + '.pkl')):
        dict_stations = read_dict(path, filename + '.pkl')

        # check if gamman already exists in dict and otherwise calculate
        exist = all(['gamman' in list(dict_stations[station].keys()) for station in dict_stations.keys()[:-1]])
        if exist:
            print('File already exists including gamman variable')
        else:
            for station in dict_stations.keys()[:-1]:
                #TODO: NORMAL TEMPERATURE OR CONSERVATIVE TEMPERATURE????
                dict_stations[station]['gamman'] = nds.gamma_n(np.nan_to_num(dict_stations[station]['SA']), np.nan_to_num(dict_stations[station]['T']),
                                                                dict_stations['P'], dict_stations['P'].size,
                                                                dict_stations[station]['lon'], dict_stations[station]['lat'])[0]
                nans = np.where(dict_stations[station]['gamman'] == -99)
                dict_stations[station]['gamman'][nans] = np.nan

            # write dictionary to file
            # reload(sys)
            # sys.setdefaultencoding('utf8')
            # print(sys.getdefaultencoding())
            write_dict(dict_stations, path, filename)
            print('data stored in file')
    else:
        print('No file with dictionary exists, please run this script in python3 to create TEOS-10 variables')