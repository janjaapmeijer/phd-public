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
gx = mlab.pipeline.grid_plane(d)
gy = mlab.pipeline.grid_plane(d)
gy.grid_plane.axis = 'y'
gz = mlab.pipeline.grid_plane(d)
gz.grid_plane.axis = 'z'
iso = mlab.pipeline.iso_surface(d)
iso.contour.maximum_contour = 75.0
vec = mlab.pipeline.vectors(d)
vec.glyph.mask_input_points = True
vec.glyph.glyph.scale_factor = 1.5

mlab.show()
