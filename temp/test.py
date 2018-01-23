for i, method in enumerate(('nearest', 'linear', 'cubic')):
    Ti = griddata((px, py), f(px,py), (X, Y), method=method)
    r, c = (i+1) // 2, (i+1) % 2
    ax[r,c].contourf(X, Y, Ti)
    ax[r,c].set_title('method = {}'.format(method))


zz = np.array([dict_trans[transect]['pt'], dict_trans[transect]['SA'], dict_trans[transect]['spiciness0'], dict_trans[transect]['sigma0']])
types = np.chararray(cmaps.shape, itemsize=4)
types[:], types[-1, -1] = 'pcol', 'conf'

def subp_figure(x, y, zz, cmaps=None, xyz=None, levels=None, ct_levels=None, titles=None, xlab=None, ylab=None, stitle=None, types=None, filename=None):

    from matplotlib.colors import BoundaryNorm, Normalize
    import matplotlib.cm as cm

    fig, ax = plt.subplots()

    for i in


    empty = np.empty(zz.shape[0:2], dtype='object')
    if types is None:
        types = empty.copy()
        types[:] = b'pcol'
    if cmaps is None:
        cmaps = empty.copy()
        cmaps[:] = cm.jet
    if titles is None:
        titles = empty.copy()


    row, col = types.shape
    fig, ax = plt.subplots(row, col, sharex='col', sharey='row', figsize=(20, 8))
    ax.shape = types.shape
    for r in range(row):
        for c in range(col):
            if types[r, c] == b'conf':
                cs = ax[r, c].contourf(x, y, zz[r, c], cmap=cmaps[r, c]) if levels is None else \
                    ax[r, c].contourf(x, y, zz[r, c], levels[r, c], cmap=cmaps[r, c])
                if xyz is not None:
                    ct = ax[r, c].scatter(xyz[0], xyz[1], c=xyz[2], cmap=cmaps[r, c], edgecolors='k')
            else:
                norm = Normalize(vmin=np.nanmin(zz[r, c]), vmax=np.nanmax(zz[r, c])) if levels is None else \
                    BoundaryNorm(levels[r, c], ncolors=cmaps[r, c].N)
                cs = ax[r, c].pcolor(x, y, zz[r, c], cmap=cmaps[r, c], norm=norm)
                # if xyz is not None:
                #     ct = ax[row-1, col-1].contour(xyz[0], xyz[1], xyz[2], ct_levels, colors='k')
                #     ax[row-1, col-1].clabel(ct, ct_levels[0::2], fontsize=8, use_clabeltext=True)
                #     for ib, b in enumerate(ct_levels):
                #         zc = ct.collections[ib]
                #         plt.setp(zc, linewidth=1)
            fig.colorbar(cs, ax=ax[r, c])
            ax[r, c].set_title(titles[r, c])
            ax[row-1, c].set_xlabel(xlab)
        ax[r, c].set_xlim([x.min(), x.max()])
        ax[r, 0].set_ylabel(ylab)
        # ax[r, 0].invert_yaxis()
    plt.suptitle(stitle)
    if filename:
        fig.savefig(filename, transparent=True)
    # fig.canvas.manager.window.attributes('-topmost', 1)
    # fig.canvas.manager.window.attributes('-topmost', 0)
    return fig, ax

from numpy import mgrid, empty, sin, pi
from tvtk.api import tvtk
from mayavi import mlab

# Generate some points.
x, y, z = mgrid[1:6:11j, 0:4:13j, 0:3:6j]
base = x[..., 0] + y[..., 0]
# Some interesting z values.
for i in range(z.shape[2]):
    z[..., i] = base * 0.25 * i

# The actual points.
pts = empty(z.shape + (3,), dtype=float)
pts[..., 0] = x
pts[..., 1] = y
pts[..., 2] = z

# Simple scalars.
scalars = x * x + y * y + z * z
# Some vectors
vectors = empty(z.shape + (3,), dtype=float)
vectors[..., 0] = (4 - y * 2)
vectors[..., 1] = (x * 3 - 12)
vectors[..., 2] = sin(z * pi)

# We reorder the points, scalars and vectors so this is as per VTK's
# requirement of x first, y next and z last.
pts = pts.transpose(2, 1, 0, 3).copy()
pts.shape = int(pts.size / 3), 3
scalars = scalars.T.copy()
vectors = vectors.transpose(2, 1, 0, 3).copy()
vectors.shape = int(vectors.size / 3), 3

# Create the dataset.
sg = tvtk.StructuredGrid(dimensions=x.shape, points=pts)
sg.point_data.scalars = scalars.ravel()
sg.point_data.scalars.name = 'temperature'
sg.point_data.vectors = vectors
sg.point_data.vectors.name = 'velocity'

# Thats it!

# Now visualize the data.
d = mlab.pipeline.add_dataset(sg)
# gx = mlab.pipeline.grid_plane(d)
# gy = mlab.pipeline.grid_plane(d)
# gy.grid_plane.axis = 'y'
# gz = mlab.pipeline.grid_plane(d)
# gz.grid_plane.axis = 'z'
iso = mlab.pipeline.iso_surface(d)
iso.contour.maximum_contour = 75.0
vec = mlab.pipeline.vectors(d)
vec.glyph.mask_input_points = True
vec.glyph.glyph.scale_factor = 1.5

mlab.show()

###################
# https://stackoverflow.com/questions/9419451/3d-contour-plot-from-data-using-mayavi-python

from scipy.interpolate import griddata
import numpy as np

# Create some test data, 3D gaussian, 200 points
dx, pts = 2, 100j

N = 500
R = np.random.random((N,3))*2*dx - dx
V = np.exp(-( (R**2).sum(axis=1)) )

# Create the grid to interpolate on
X,Y,Z = np.mgrid[-dx:dx:pts, -dx:dx:pts, -dx:dx:pts]

# Interpolate the data
F = griddata(R, V, (X,Y,Z))


from mayavi.mlab import *
contour3d(F,contours=8,opacity=.2 )


import numpy as np
from tvtk.api import tvtk
from mayavi import mlab

points = np.random.normal(0, 1, (1000, 3))
ug = tvtk.UnstructuredGrid(points=points)
ug.point_data.scalars = np.sqrt(np.sum(points**2, axis=1))
ug.point_data.scalars.name = "value"
ds = mlab.pipeline.add_dataset(ug)
delaunay = mlab.pipeline.delaunay3d(ds)
iso = mlab.pipeline.iso_surface(delaunay)
iso.actor.property.opacity = 0.1
iso.contour.number_of_contours = 10
mlab.show()


###################

import pyvtk
import numpy as np
from scipy.spatial import Delaunay
from mayavi import mlab

# Generate the random points
npoints = 1000
np.random.seed(1)
x = np.random.normal(size=npoints)*np.pi
y = np.random.normal(size=npoints)*np.pi
z = np.sin((x))+np.cos((y))
# Generate random data
pointPressure = np.random.rand(npoints)

# Compute the 2D Delaunay triangulation in the x-y plane
xTmp=list(zip(x,y))
tri=Delaunay(xTmp)

# Generate Cell Data
nCells=tri.nsimplex
cellTemp=np.random.rand(nCells)

# Zip the point co-ordinates for the VtkData input
points=list(zip(x,y,z))

vtk = pyvtk.VtkData(pyvtk.UnstructuredGrid(points, triangle=tri.simplices),
                    pyvtk.PointData(pyvtk.Scalars(pointPressure,name='Pressure')),
                    pyvtk.CellData(pyvtk.Scalars(cellTemp,name='Temperature')),
                    '2D Delaunay Example')

src = mlab.pipeline.open(vtk)

vtk.tofile('Delaunay2D')
vtk.tofile('Delaunay2Db','binary')

# Compute the 3D Delaunay triangulation in the x-y plane
xTmp=list(zip(x,y,z))
tri=Delaunay(xTmp)

# Generate Cell Data
nCells=tri.nsimplex
cellTemp=np.random.rand(nCells)

# Zip the point co-ordinates for the VtkData input
points=list(zip(x,y,z))

vtk = pyvtk.VtkData(\
  pyvtk.UnstructuredGrid(points,
    tetra=tri.simplices
    ),
  pyvtk.PointData(pyvtk.Scalars(pointPressure,name='Pressure')),
  pyvtk.CellData(pyvtk.Scalars(cellTemp,name='Temperature')),
  '3D Delaunay Example'
)










