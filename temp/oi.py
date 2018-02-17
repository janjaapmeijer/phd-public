# http://twister.caps.ou.edu/OBAN2002/OI.pdf
# assim.chocolate.fr/ana2d.py.txt
# http://stockage.univ-brest.fr/~herbette/Data-Analysis/data-analysis-opt-interp.pdf


import matplotlib.pyplot as plt
from OceanPy.polynomials import polyfit1d, polyfit2d
import numpy as np

xg, yg = 0, 0

lon = np.array([1, 0, -0.7071])
lat = np.array([0, 1, -0.7071])

T = np.array([5, 2, 8])

n = 1 #len(grid_points) * len(vars)
p = len(T) #len(obs)
nns = (n, n)
nps = (n, p)
pns = (p, n)
pps = (p, p)

# observation vector
yo = np.matrix(T)

# linear operator matrix
H = np.matrix(np.zeros(pns))

def H(lon, lat, xg, yg, gridsize=(1, 1)):
    idim, jdim = gridsize[0], gridsize[1]

    x = idim / 2 + 1 + (lon - xg) /
# background error covariance matrix
# B = np.matrix(np.zeros(nns))


# observation error covariance matrix
# R =

# # initialise background error covariance matrix
# varb = np.var(obs)
# B = np.matrix(np.eye(nns))



xm, ym, zm = polyfit2d(lon, lat, T, order=1, gridsize=(3, 3))
xmp, ymp, zmp = polyfit2d(lon, lat, T, order=1, point=(xg, yg))

plt.pcolor(xm, ym, zm)
scat = plt.scatter(lon, lat, c=T, s=100)
plt.scatter(xg, yg, facecolors='w', s=100)
plt.axis('equal')
plt.colorbar(scat)
