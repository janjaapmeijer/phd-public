#!/usr/bin/evn oceanv3

import numpy as np
from itertools import combinations
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# x = np.array([0, 1, 2, 3])
# y = np.array([-1, 0.2, 0.9, 2.1])

def polyfit_1d(x, y, order=2, grid=True):
    '''
    Uses x, y data and fits polynomial using the least-squares method for given order.
    Solves the polynomial of the form:
    y = ax + b                              (order = 1 or 'linear')
    y = ax^2 + bx + c                       (order = 2 or 'quadratic')
    y = ax^3 + bx^2 + cx + d                (order = 3 or 'cubic'
    '''

    xm = np.linspace(x.min(), x.max()) if grid else x.copy()

    if order == 1:
        A = np.vstack([np.ones(len(x)), x]).T
        coef = np.linalg.lstsq(A, y)[0]
        # ym = coef[1] * xm + coef[0]

        # expressed in matrix notation
        ym = np.dot(np.c_[np.ones(len(xm)), xm], coef)

    elif order == 2:
        A = np.vstack([np.ones(len(x)), x, x**2]).T
        coef = np.linalg.lstsq(A, y)[0]
        # ym = coef[2] * xm**2 + coef[1] * xm + coef[0]

        # expressed in matrix notation
        ym = np.dot(np.c_[np.ones(len(xm)), xm, xm**2], coef)

    elif order == 3:
        A = np.vstack([np.ones(len(x)), x, x**2, x**3]).T
        coef = np.linalg.lstsq(A, y)[0]

        # expressed in matrix notation
        ym = np.dot(np.c_[np.ones(len(xm)), xm, xm ** 2, xm**3], coef)

    elif order == 4:
        A = np.vstack([np.ones(len(x)), x, x**2, x**3, x**4]).T
        coef = np.linalg.lstsq(A, y)[0]

        # expressed in matrix notation
        ym = np.dot(np.c_[np.ones(len(xm)), xm, xm ** 2, xm**3, xm**4], coef)

    else:
        raise ValueError('This function does not solve higher than 4th order relations')

    return xm, ym

# plt.plot(x, y, 'o', label='Original data', markersize=10)
# plt.plot(*polyfit_1d(x, y, order=1, grid=True), 'r', label='linear polynomial')
# plt.plot(*polyfit_1d(x, y, order=2, grid=True), 'g', label='quadratic polynomial')
# plt.plot(*polyfit_1d(x, y, order=3, grid=True), 'm', label='cubic polynomial')
# plt.plot(*polyfit_1d(x, y, order=4, grid=True), 'c', label='quartic polynomial')
# plt.legend()


# mean = np.array([0.0,0.0,0.0])
# cov = np.array([[1.0,-0.5,0.8], [-0.5,1.1,0.0], [0.8,0.0,1.0]])
# data = np.random.multivariate_normal(mean, cov, 50)
# x, y, z = data[:,0], data[:,1], data[:,2]


def polyfit_2d(x, y, z, order=2, gridsize=(50, 100)):
    '''
    Uses x, y and z data and fits polynomial using the least-squares method for given order.
    Solves the polynomial of the form:
    z = ax + by + c                                                     (order = 1 or 'linear')
    y = ax^2 + by^2 + cxy + dx + ey + f                                 (order = 2 or 'quadratic')
    y = ax^3 + by^3 + cxy^2 + dx^2y + ex^2 + fy^2 + gxy + hx + iy + j   (order = 3 or 'cubic'
    '''


    xm, ym = np.meshgrid(np.linspace(x.min(), x.max(), gridsize[0]),
                       np.linspace(y.min(), y.max(), gridsize[1]))
    xf, yf = xm.flatten(), ym.flatten()

    if order == 1:
        A = np.vstack([np.ones(len(x)), x, y]).T
        coef = np.linalg.lstsq(A, z)[0]

        # expressed in matrix notation
        zm = np.dot(np.c_[np.ones(len(xf)), xf, yf], coef).reshape(xm.shape)

    elif order == 2:
        A = np.vstack([np.ones(len(x)), x, y, x * y, x**2, y**2]).T
        coef = np.linalg.lstsq(A, z)[0]

        # expressed in matrix notation
        zm = np.dot(np.c_[np.ones(len(xf)), xf, yf, xf * yf, xf**2, yf**2], coef).reshape(xm.shape)

    elif order == 3:
        A = np.vstack([np.ones(len(x)), x, y, x * y, x**2, y**2, x**2 * y, x * y**2, x**3, y**3]).T
        coef = np.linalg.lstsq(A, z)[0]

        # expressed in matrix notation
        zm = np.dot(np.c_[np.ones(len(xf)), xf, yf, xf*yf, xf**2, yf**2, xf**2 * yf, xf * yf**2, xf**3, yf**3],
                    coef).reshape(xm.shape)

    elif order == 4:
        A = np.vstack([np.ones(len(x)), x, y, x*y, x**2, y**2, x**2 * y, x * y**2, x**3, y**3,
                       x ** 2 * y ** 2, x**3 * y, x * y**3, x**4, y**4]).T
        coef = np.linalg.lstsq(A, z)[0]

        # expressed in matrix notation
        zm = np.dot(np.c_[np.ones(len(xf)), xf, yf, xf*yf, xf**2, yf**2, xf**2 * yf, xf * yf**2, xf**3, yf**3,
                          xf ** 2 * yf ** 2, xf**3 * yf, xf * yf**3, xf**4, yf**4],
                    coef).reshape(xm.shape)

    else:
        raise ValueError('This function does not solve higher than 4th order relations')


    return xm, ym, zm

import itertools


mean = np.array([0.0,0.0,0.0])
cov = np.array([[1.0,-0.5,0.8], [-0.5,1.1,0.0], [0.8,0.0,1.0]])
data = np.random.multivariate_normal(mean, cov, 50)
x, y, z = data[:,0], data[:,1], data[:,2]

# x = np.array([1.2, 1.3, 1.6, 2.5, 2.3, 2.8])
# y = np.array([167.0, 180.3, 177.8, 160.4, 179.6, 154.3])
# z = np.array([-0.3, -0.8, -0.75, -1.21, -1.65, -0.68])

# POLYNOMIAL FIT FUNCTION
def polyfit2d(x, y, z, order=2, gridsize=(50, 100)):
    ''' built in residuals and RMSE '''

    xm, ym = np.meshgrid(np.linspace(x.min(), x.max(), gridsize[0]),
                       np.linspace(y.min(), y.max(), gridsize[1]))
    xf, yf = xm.flatten(), ym.flatten()

    nterms = (order**2 + 3*order + 2)/2

    P = np.zeros((x.size, nterms))
    # ij = itertools.product(range(order+1), range(order+1))

    ij = []
    for i in range(0, order+1):
        for j in range(0, order+1):
            if i + j <= order:
                ij.append((i,j))


    for k, (i,j) in enumerate(ij):
        P[:, k] = x**i * y**j
    A = np.linalg.lstsq(P, z)[0]

    zf = np.zeros(xf.shape)
    for alpha, (i,j) in zip(A, ij):
        zf += alpha * xf**i * yf**j

    zm = zf.reshape(xm.shape)

    return xm, ym, zm



# zz = polyfit2d(x, y, z, order=2)






# plt.plot(x, y, 'o', label='Original data', markersize=10)
# plt.plot(*polyfit_1d(x, y, order=1, grid=True), 'r', label='linear polynomial')
# plt.plot(*polyfit_1d(x, y, order=2, grid=True), 'g', label='quadratic polynomial')
# plt.plot(*polyfit_1d(x, y, order=3, grid=True), 'm', label='cubic polynomial')
# plt.plot(*polyfit_1d(x, y, order=5, grid=True), 'c', label='quartic polynomial')
# plt.legend()


# plot points and fitted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(*polyfit2d(x, y, z, order=2), rstride=1, cstride=1, alpha=0.2)
ax.plot(x, y, z, 'o', label='Original data', markersize=10)
# ax.scatter(x, y, z, c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
ax.set_zlabel('Z')
ax.axis('equal')
ax.axis('tight')
plt.show()

