from mayavi import mlab
import numpy as np

### EXAMPLE 1
x = np.random.random(10000)
y = np.random.random(10000)
z = np.random.random(10000)
s = x*x + 0.5*y*y + 3*z*z
src = mlab.pipeline.scalar_scatter(x, y, z, s)

g = mlab.pipeline.glyph(src, mode='point')
gs = mlab.pipeline.gaussian_splatter(src)
gs.filter.radius = 0.25
o = mlab.pipeline.outline(gs)
cp = mlab.pipeline.scalar_cut_plane(gs)
iso=mlab.pipeline.iso_surface(gs)
mlab.show()

### EXAMPLE 2
x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
s = np.sin(x*y*z)/(x*y*z)

src = mlab.pipeline.scalar_field(s)
mlab.pipeline.iso_surface(src, contours=[s.min()+0.1*s.ptp(), ], opacity=0.3)
mlab.pipeline.iso_surface(src, contours=[s.max()-0.1*s.ptp(), ],)

mlab.show()

### EXAMPLE 3
# Create data with x and y random in the [-2, 2] segment, and z a
# Gaussian function of x and y.
np.random.seed(12345)
x = 4 * (np.random.random(500) - 0.5)
y = 4 * (np.random.random(500) - 0.5)

def f(x, y):
    return np.exp(-(x ** 2 + y ** 2))
z = f(x, y)

mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# Visualize the points
pts = mlab.points3d(x, y, z, z, scale_mode='none', scale_factor=0.2)
# Create and visualize the mesh
mesh = mlab.pipeline.delaunay2d(pts)
surf = mlab.pipeline.surface(mesh)
mlab.view(47, 57, 8.2, (0.1, 0.15, 0.14))
mlab.show()

### EXAMPLE 4
### surface with spheres
x,y = np.meshgrid(np.linspace(-2.5,2), np.linspace(-2,2))
f = lambda x,y: .4*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
z=f(x,y)
mlab.surf(x.T,y.T,z.T, colormap="copper")

px,py = np.meshgrid(np.arange(-2,2)+.25, np.arange(-2,2)+.75)
px,py = px.flatten(),py.flatten()
pz = np.ones_like(px)*0.05
r = np.ones_like(px)*.4
mlab.points3d(px,py,pz,r, color=(0.9,0.05,.3), scale_factor=1)
mlab.show()