from _setup import *

from netCDF4 import Dataset
import numpy as np
from OceanPy.animations import play2D



import matplotlib.pyplot as plt
import matplotlib.patches as patches



# altimetry
input_file = os.path.join(datadir, 'external', 'altimetry', 'geo_vel_from_SSH.nc')
ssh = Dataset(input_file)



fig, ax = plt.subplots()
ax.pcolor(ssh['LONGITUDE'][:], ssh['LATITUDE'][:], ssh['UCUR'][0,::])





play2D(ssh['LONGITUDE'][:], ssh['LATITUDE'][:], ssh['UCUR'][:], cmin=0, cmax=0.6)




# BRAN/ OFAM
urlu = 'http://dapds00.nci.org.au/thredds/dodsC/gb6/BRAN/BRAN_2016/OFAM/ocean_u_1998_03.nc'
urlv = 'http://dapds00.nci.org.au/thredds/dodsC/gb6/BRAN/BRAN_2016/OFAM/ocean_v_1998_03.nc'

branu = Dataset(urlu)
branv = Dataset(urlv)

time = branu['Time']
depth = branu['st_ocean']
xu = branu['xu_ocean'][1374:1439]
yu = branu['yu_ocean'][229:264]
u = branu['u'][:, :, 229:264, 1374:1439]
v = branv['v'][:, :, 229:264, 1374:1439]

# calculate speed
V = np.ma.masked_all(u.shape)
for t in range(len(time)):
    for d in range(len(depth)):
        V[t, d] = np.sqrt(u[t, d]**2 + v[t,d]**2)

i300 = min(enumerate(depth), key=lambda x: abs(x[1]-300))[0]

xgrd, ygrd = np.meshgrid(xu, yu)

Vmmean = np.mean(V,axis=0)

fig, ax = plt.subplots()
ax.pcolor(xgrd, ygrd, Vmmean[0])
ax.set_aspect('equal')

plt.fill([138, 143.5, 143.5, 138], [-51.75, -51.75, -48.75, -48.75],
         'b', facecolor='none', edgecolor='k', linewidth=2)

branu.close()
branv.close()

