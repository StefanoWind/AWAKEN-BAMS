# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:15:55 2024

@author: sletizia
"""

# -*- coding: utf-8 -*-
"""
Wake analysis
"""
import os
cd=os.path.dirname(__file__)
import sys

import numpy as np
import glob
import yaml
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
import re
from mayavi.mlab import *
from matplotlib.patches import Circle

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

#%% Inputs
source_config=os.path.join(cd,'config.yaml')
source_log='data/20230101.000500-20240101.224500.awaken.glob.summary.csv'
source_stats=os.path.join(cd,'data/awaken/rt1.lidar.z02.a0/5.8.135.225.0.5/5.8.135.225.0.5.stats.nc')


#site
D=127#[m] rotor diameter
H=90#[m] hub height

#stats
max_err_u=0.2#maximum error in mean normalized streamwise velocity
max_err_TI=4#maximum error in mean turbulence intensity

#graphics
#plot area[D]
ymin=-2
ymax=2
zmin=-1
zmax=2

x_plot=[3,4.5,6,8]
r_free=0.75

#%% Initialization
Data=xr.open_dataset(source_stats)

x0=Data.x.values
y0=Data.y.values
z0=Data.z.values
X0,Y0,Z0=np.meshgrid(x0,y0,z0,indexing='ij')

Data['u_avg_qc']= Data['u_avg'].where(Data['u_top']-Data['u_low']<max_err_u)

#extend for better graphics
dy=np.diff(Data.y.values)[0]
new_y = np.arange(-100*dy,100*dy,dy/2)
dz=np.diff(Data.z.values)[0]
new_z = np.arange(-50*dz,50*dz,dz)/2
u_avg_ext=Data['u_avg_qc'].interp(y=new_y).interp(z=new_z)

x=u_avg_ext.x.values
y=u_avg_ext.y.values
z=u_avg_ext.z.values

X,Y,Z=np.meshgrid(x,y,z,indexing='ij')


free=(u_avg_ext.y**2+u_avg_ext.z**2)**0.5/D>r_free
u_free=u_avg_ext.where(free).mean(dim='x').mean(dim='y')
vd_avg=1-u_avg_ext/u_free


#%% Plots
# figure(size=(800, 600),bgcolor=(1, 1, 1))
    

# contour3d(X, Y, Z,u_avg, contours=[0.6], opacity=1, colormap='coolwarm')
# contour3d(X, Y, Z,u_avg, contours=[0.75], opacity=0.75, colormap='coolwarm')


# #velocity (non modified)
# figure(size=(800, 600),bgcolor=(1, 1, 1))
# for xp in x_plot:
#     i=np.where(xp==x/D)[0][0]
#     obj=volume_slice(X0,Y0,Z0,Data['u_avg_qc'],plane_orientation='x_axes', slice_index=i,colormap='coolwarm',vmin=0.5,vmax=1.2)
#     obj.module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
#     view(azimuth=240, elevation=70, distance=1000)

# #velocity
# figure(size=(800, 600),bgcolor=(1, 1, 1))
# for xp in x_plot:
#     i=np.where(xp==x/D)[0][0]
#     obj=volume_slice(X,Y,Z,u_avg_ext,plane_orientation='x_axes', slice_index=i,colormap='coolwarm',vmin=0.5,vmax=1.4)
#     obj.module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
#     view(azimuth=240, elevation=70, distance=1000)


# #freestream velocity
# figure(size=(800, 600),bgcolor=(1, 1, 1))
# for xp in x_plot:
#     i=np.where(xp==x/D)[0][0]
#     obj=volume_slice(X,Y,Z,u_avg_ext.where(free),plane_orientation='x_axes', slice_index=i,colormap='coolwarm',vmin=0.5,vmax=1.2)
#     obj.module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
#     view(azimuth=240, elevation=70, distance=1000)

# #velocity deficit
# figure(size=(800, 600),bgcolor=(1, 1, 1))
# for xp in x_plot:
#     i=np.where(xp==x/D)[0][0]
#     obj=volume_slice(X,Y,Z,vd_avg,plane_orientation='x_axes', slice_index=i,colormap='inferno',vmin=-0.1,vmax=0.5)
#     obj.module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
#     view(azimuth=240, elevation=70, distance=1000)

# #using surface
Y,Z=np.meshgrid(y0,z0)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for xp in x_plot:
    i=np.where(xp==x/D)[0][0]
    ax.contourf(Data['u_avg_qc'].values[i,:,:].T, Y, Z,zdir='x',offset=xp,levels=np.arange(0.6,1.5,0.1))

ax.set_xlim([0, 10])
# ax.set_ylim([, 5])
# ax.set_zlim([-5, 5])