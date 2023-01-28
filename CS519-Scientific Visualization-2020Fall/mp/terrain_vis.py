from pyvista import set_plot_theme
set_plot_theme("document")

import pyvista as pv
import math
import numpy as np
import pylab as plt
from pyvista import examples

dem = examples.download_crater_topo()
subset = dem.extract_subset((572, 828, 472, 728, 0, 0), (1,1,1))
# pv.plot_itk(subset)
terrain = subset.warp_by_scalar()
# pv.plot_itk(terrain)

# xrng = np.arange(-10, 10, 2)                # [-10,  -8,  -6,  -4,  -2,   0,   2,   4,   6,   8]
# yrng = np.arange(-10, 10, 2)
# zrng = np.arange(-10, 10, 2)
# x_example, y_example, z_example = np.meshgrid(xrng, yrng, zrng)
# grid_example = pv.StructuredGrid(x_example, y_example, z_example)
# print(x_example, y_example, z_example)
terrain_x = terrain.x
terrain_y = terrain.y
terrain_z = terrain.z
f = 2
xnew = terrain_x[::f, ::f, ]
ynew = terrain_y[::f, ::f, ]
znew = terrain_z[::f, ::f, ]
coarse = pv.StructuredGrid(xnew, ynew, znew)
# coarse['values'] = pv.plotting.normalize(coarse.z.flatten("F"))
# pv.plot_itk(coarse, scalars='values')

def bilin(x,y,points):
    # sorted from ymin to ymax
    st_points = sorted(points, key=lambda x: x[1])
    xmin = st_points[0][0]
    ymin = st_points[0][1]
    xmax = st_points[1][0]
    ymax = st_points[-1][1]
    x_r = (x-xmin)/(xmax-xmin)
    y_r = (y-ymin)/(ymax-ymin)
    val_b = (1-x_r)*st_points[0][2] + x_r*(st_points[1][2])
    val_t = (1-x_r)*st_points[2][2] + x_r*(st_points[3][2])
    val_c = (1-y_r)*val_b + y_r*val_t
    return val_c

errz = []
intz = np.zeros_like(terrain.z)
xlen   = coarse.z.shape[0]-1   #Number of cells (points-1) on the x-axis of the coarse mesh
ylen   = coarse.z.shape[1]-1   #Number of cells (points-1) on the y-axis of the coarse mesh
scale = int((terrain.z.shape[0]-1)/(coarse.z.shape[0]-1)) #Reduction factor between original and coarse; should equal 2

def mesh_bilin(mesh, scale):
    scale = int(scale)
    mesh_shape = mesh.shape  
    init_size_x = int((mesh_shape[0]-1) * scale + 1)
    init_size_y = int((mesh_shape[1]-1) * scale + 1)
    new_mesh = np.zeros((init_size_x, init_size_y, 1))
    for i in range(mesh_shape[0]-1):
        for j in range(mesh_shape[1]-1):
            rect_z = mesh[i:i+2,j:j+2]
            rect_v = [(i, j, rect_z[0][0][0]), 
                      (i+1,j,rect_z[1][0][0]),
                      (i,j+1,rect_z[0][1][0]),
                      (i+1,j+1,rect_z[1][1][0])]
            new_x = np.linspace(i, i+1, scale+1)
            new_y = np.linspace(j, j+1, scale+1)
            for n, x in enumerate(new_x):
                for m, y in enumerate(new_y):
                    new_mesh[i*scale+n][j*scale+m][0] = bilin(x, y, rect_v)
    return new_mesh

intz = mesh_bilin(coarse.z, scale)
intx = mesh_bilin(coarse.x, scale)
inty = mesh_bilin(coarse.y, scale)
errz = np.ravel(abs(intz-terrain_z))
intmesh = pv.StructuredGrid(intx, inty, intz)
intmesh['errors'] = pv.plotting.normalize(errz)


print(errz[-10])

