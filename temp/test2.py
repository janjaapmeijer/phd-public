from _setup import *

from netCDF4 import Dataset

import numpy as np

import matplotlib.pyplot as plt

output_file = os.path.join(datadir, 'processed', 'ss9802', 'netcdf', 'ss9802_ctd_gsw.nc')

nc = Dataset(output_file, 'r')

sigma0 = nc['sigma0'][0, ]
z = nc['z'][0, ]

# The reference depth is set at 10 m to avoid a large part of the strong diurnal cycle in the top few meters of the ocean.
i10, ref_dep = min(enumerate(z), key=lambda x: abs(x[1]+10))
m = len(sigma0[i10:])

# The fixed criterion in density is 0.03 kg/m3 difference from surface
for i in range(0, m):
    if abs(sigma0[i10] - sigma0[i10:]) > 0.03:
        imld = i
        print('reference depth: ', ref_dep,
              'mixed layer depth: ', z[i10 + imld])
        break







# plt.plot(sigma0, z)
# intv = np.linspace(26.4, 27.4, 6)

# def layer_thickness(sigma0, z, intv=intv, threshold=0.01):

threshold=0.005
smin = sigma0.min()
smax = sigma0.max()

for int in intv:
    for i, s in enumerate(sigma0):
        if abs(s - int) < threshold:
            print(int, s, i)

min(enumerate(sigma0), key=lambda x: abs(x[1]-intv[0]))
