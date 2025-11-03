# -*- coding: utf-8 -*-
"""
Plot 3D RHIa
"""

import os
cd=os.path.dirname(__file__)
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500
plt.close('all')

#%% Inputs
source_layout='data/20250225_AWAKEN_layout.nc'

site_ref='H05'
farm_sel='King Plains'
turb_sel=['H05','G02','F04','E06']

xlim=[-2000,5000]
ylim=[-2000,8000]
zlim=[0,1000]

#%% Functions

def draw_turbine_3d(ax,x,y,z,D,H,yaw):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # Correct import
    from stl import mesh  # Correct import for Mesh
    
    # Load the STL file of the 3D turbine model
    turbine_mesh = mesh.Mesh.from_file(os.path.join(cd,'blades.stl'))
    tower_mesh = mesh.Mesh.from_file(os.path.join(cd,'tower.stl'))
    nacelle_mesh = mesh.Mesh.from_file(os.path.join(cd,'nacelle.stl'))

    #translate
    translation_vector = np.array([-125, -110, -40])
    turbine_mesh.vectors += translation_vector

    translation_vector = np.array([-125, -95, -150])
    tower_mesh.vectors += translation_vector

    translation_vector = np.array([-125, -100,-10])
    nacelle_mesh.vectors += translation_vector

    #rescale
    scaling_factor = 1/175*D
    turbine_mesh.vectors *= scaling_factor

    scaling_factor = 1/250*D
    scaling_factor_z=1/0.6*H/D
    tower_mesh.vectors *= scaling_factor
    tower_mesh.vectors[:, :, 2] *= scaling_factor_z

    scaling_factor = 1/175*D
    nacelle_mesh.vectors *= scaling_factor

    #rotate
    theta = np.radians(180+yaw)  
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,             0,             1]
    ])

    turbine_mesh.vectors = np.dot(turbine_mesh.vectors, rotation_matrix)
    tower_mesh.vectors = np.dot(tower_mesh.vectors, rotation_matrix)
    nacelle_mesh.vectors = np.dot(nacelle_mesh.vectors, rotation_matrix)

    #translate
    translation_vector = np.array([x, y, z])
    turbine_mesh.vectors += translation_vector
    tower_mesh.vectors += translation_vector
    nacelle_mesh.vectors += translation_vector


    # Extract the vertices from the rotated STL mesh
    faces = turbine_mesh.vectors  # Each face of the mesh

    # Create a Poly3DCollection for the faces
    poly_collection = Poly3DCollection(faces, facecolors=(0,0,0,0.8), linewidths=1, edgecolors=None, alpha=0.5)
    ax.add_collection3d(poly_collection)

    # Get the scale from the mesh to fit it properly
    scale = np.concatenate([turbine_mesh.points.min(axis=0), turbine_mesh.points.max(axis=0)])

    # Extract the vertices from the rotated STL mesh
    faces = tower_mesh.vectors  # Each face of the mesh

    # Create a Poly3DCollection for the faces
    poly_collection = Poly3DCollection(faces, facecolors=(0,0,0,0.8), linewidths=1, edgecolors=None, alpha=0.5)
    ax.add_collection3d(poly_collection)

    # Extract the vertices from the rotated STL mesh
    faces = nacelle_mesh.vectors  # Each face of the mesh

    # Create a Poly3DCollection for the faces
    poly_collection = Poly3DCollection(faces, facecolors=(0,0,0,0.8), linewidths=1, edgecolors=None, alpha=0.5)
    ax.add_collection3d(poly_collection)


    # Set the scale for the axis
    ax.auto_scale_xyz(scale, scale, scale)

#%% Initialization
Turbines=xr.open_dataset(source_layout,group='turbines')
Turbines=Turbines.where(Turbines['Wind plant']==farm_sel,drop=True)
x_ref=Turbines.x_utm[Turbines.name==site_ref].values[0]
y_ref=Turbines.y_utm[Turbines.name==site_ref].values[0]

#%% Plots
plt.close('all')
fig=plt.figure(figsize=(18,10))
ax = fig.add_subplot(111, projection='3d')

for x,y,D,H in zip(Turbines.x_utm.values-x_ref,
                   Turbines.y_utm.values-y_ref,
                   Turbines.Diameter.values,
                   Turbines['Hub height'].values):
    if x>xlim[0] and x<xlim[1] and y>ylim[0] and y<ylim[1]:
        draw_turbine_3d(ax, x, y, H*2, D*2, H*2, 180)

E,R=np.meshgrid(np.arange(0,181,3),np.arange(0,1000,10))
x_rhi=R*0
y_rhi=np.cos(np.radians(E))*R
z_rhi=np.sin(np.radians(E))*R

for t in turb_sel:
    plt.plot(x_rhi+Turbines.x_utm[Turbines.name==t].values[0]-x_ref,
             y_rhi+Turbines.y_utm[Turbines.name==t].values[0]-y_ref,
             z_rhi+Turbines['Hub height'][Turbines.name==t].values[0],'.g',alpha=0.1,markersize=1)
    plt.plot(x_rhi[-1,:]+Turbines.x_utm[Turbines.name==t].values[0]-x_ref,
             y_rhi[-1,:]+Turbines.y_utm[Turbines.name==t].values[0]-y_ref,
             z_rhi[-1,:]+Turbines['Hub height'][Turbines.name==t].values[0],'.g',alpha=1,markersize=1)            

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.axis("off")
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)
ax.set_aspect('equal')