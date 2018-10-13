from _setup import *

from netCDF4 import Dataset, num2date
import numpy as np

from OceanPy.interpolation import OI, polyfit2d

import matplotlib.pyplot as plt

input_file_adcp = os.path.join(datadir, 'processed', 'ss9802', 'netcdf', 'ss9802_adcp_ctd.nc')
input_file_ssh = os.path.join(datadir, 'external', 'ssh', 'SSH_OceanCurrent_137.5_-52_144_-48.5_199803.nc')
input_file_ctd = os.path.join(datadir, 'processed', 'ss9802', 'netcdf', 'ss9802_ctd_gsw.nc')

adcp = Dataset(input_file_adcp)
ssh = Dataset(input_file_ssh)
ctd = Dataset(input_file_ctd)

loni, lati = ssh['LONGITUDE'][:], ssh['LATITUDE'][:]

# Vg_ssh = np.ma.masked_all(ssh['UCUR'].shape)
# for t in range(ssh.dimensions['TIME'].size):
#     Vg_ssh[t,] = np.sqrt(ssh['UCUR'][t,]**2 + ssh['VCUR'][t,]**2)

# monthly mean ssh
# Vmmean = np.mean(Vg_ssh, axis=0)
ummean = np.mean(ssh['UCUR'][:], axis=0)
vmmean = np.mean(ssh['VCUR'][:], axis=0)
# interpolate ctd data
# lon, lat = ctd['lonv'][:], ctd['latv'][:]
# xx, yy, Vg_b, Vg_a = OI(lon, lat, ctd['Vg'][:,10], Lx=0.5, Ly=0.5, xx=lon_grd, yy=lat_grd, bg_fld=Vmmean)

# interpolate adcp data
lon, lat = adcp['lon_ctd'][2:], adcp['lat_ctd'][2:]
ua = np.mean(adcp['u_ctd'][2:,:11], axis=1)
va = np.mean(adcp['v_ctd'][2:,:11], axis=1)

mask = np.ma.masked_invalid(ua).mask
xx, yy, u_b, u_a = OI(lon[~mask], lat[~mask], ua[~mask],
                      Lx=(lon.max() - lon.min())/12, Ly=(lat.max() - lat.min())/14,
                      xx=loni, yy=lati, bg_fld=ummean)
v_b, v_a = OI(lon[~mask], lat[~mask], va[~mask],
              Lx=(lon.max() - lon.min())/12, Ly=(lat.max() - lat.min())/14,
              xx=loni, yy=lati, bg_fld=vmmean)[2:]

vmin, vmax = 0, 0.6
fig, ax = plt.subplots()
pcol = ax.pcolor(xx, yy, u_b, vmin=vmin, vmax=vmax)
plt.colorbar(pcol)
scat = ax.scatter(lon, lat, c=ua, s=100, vmin=vmin, vmax=vmax)
plt.colorbar(scat)

fig, ax = plt.subplots()
pcol = ax.pcolor(xx, yy, u_a, vmin=vmin, vmax=vmax)
plt.colorbar(pcol)
scat = ax.scatter(lon, lat, c=ua, s=100, vmin=vmin, vmax=vmax)
plt.colorbar(scat)

vmin, vmax = -0.45, 0.45
fig, ax = plt.subplots()
pcol = ax.pcolor(xx, yy, v_b, vmin=vmin, vmax=vmax)
plt.colorbar(pcol)
scat = ax.scatter(lon, lat, c=va, s=100, vmin=vmin, vmax=vmax)
plt.colorbar(scat)

fig, ax = plt.subplots()
pcol = ax.pcolor(xx, yy, v_a, vmin=vmin, vmax=vmax)
plt.colorbar(pcol)
scat = ax.scatter(lon, lat, c=va, s=100, vmin=vmin, vmax=vmax)
plt.colorbar(scat)

