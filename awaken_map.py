# -*- coding: utf-8 -*-
"""
Plot site map with beam schematic
"""

import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('utils')
import utils as utl
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
import matplotlib
from scipy.interpolate import RegularGridInterpolator

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['savefig.dpi']=500
plt.close('all')

#%% Inputs
source_layout='data/20250225_AWAKEN_layout.nc'

site_ref='A1'
ar=5

k_wake=0.1#[m] wake expansion factor
ct=0.8
wake_lenght=5#[D] wake length
wake_iso=[0.75,0.5,0.25]#isocontour values

terrain_loc=[0,0]
terrain_dir=180
terrain_length=5000

#graphics
xlim=[-33000,5000]
ylim=[-17000,14000]

#%% Initialization
Turbines=xr.open_dataset(source_layout,group='turbines')
Topo=xr.open_dataset(source_layout,group='topography')
Sites=xr.open_dataset(source_layout,group='ground_sites')

x_ref=float(Sites.x_utm.sel(site=site_ref))
y_ref=float(Sites.y_utm.sel(site=site_ref))

X_topo,Y_topo=np.meshgrid(Topo['x_utm'].values-x_ref,Topo['y_utm'].values-y_ref)
Z_topo=np.ma.masked_invalid(Topo['elevation'].values).T

sel_x=(Topo['x_utm'].values-x_ref>xlim[-0])*(Topo['x_utm'].values-x_ref<xlim[1])
sel_y=(Topo['y_utm'].values-y_ref>ylim[-0])*(Topo['y_utm'].values-y_ref<ylim[1])

x_terrain=terrain_loc[0]+np.cos(np.radians(270-terrain_dir))*np.arange(-terrain_length/2,terrain_length/2+1,10)
y_terrain=terrain_loc[0]+np.sin(np.radians(270-terrain_dir))*np.arange(-terrain_length/2,terrain_length/2+1,10)

interp_func = RegularGridInterpolator(
    (Topo['x_utm'].values, Topo['y_utm'].values),  # grid axes
    Topo.elevation.values,                        # data on the grid
    bounds_error=False,             # allow extrapolation (or use True to raise an error)
    fill_value=np.nan               # what to return outside bounds
)

# Combine target coordinates into (N, 2) array
points = np.column_stack((x_terrain+x_ref, y_terrain+y_ref))  # shape: (N, 2)

# Evaluate interpolated values
z_terrain = interp_func(points)  # shape: (N,)
z_terrain=np.min(z_terrain)+(z_terrain-np.min(z_terrain))*10

r_terrain=(x_terrain**2+y_terrain**2)**0.5*np.sign(y_terrain)
X_terrain,Z_terrain=np.meshgrid(r_terrain,np.arange(2000)+np.min(z_terrain))
z_terrain_rep=np.tile(z_terrain,(2000,1))
ws_terrain=500**2-(Z_terrain-z_terrain_rep-500)**2+(Z_terrain-z_terrain_rep)**1.9+np.sin(np.radians(X_terrain/100))*10000
ws_terrain=ws_terrain/np.nanmax(ws_terrain)*15
ws_terrain[ws_terrain<0]=np.nan

cmap = plt.get_cmap('coolwarm')
colors_wake = [cmap(i / len(wake_iso)) for i in range(len(wake_iso))]

#%% Plots
plt.close('all')
fig=plt.figure(figsize=(18,8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X_topo[sel_y,:][:,sel_x], Y_topo[sel_y,:][:,sel_x], Z_topo[sel_y,:][:,sel_x],cmap='summer',linewidth=0, antialiased=False,alpha=1)

for x,y in zip(Sites.x_utm.values-x_ref, Sites.y_utm.values-y_ref):
    if x>xlim[0] and x<xlim[1] and y>ylim[0] and y<ylim[1]:
        z=Topo.elevation.interp(x_utm=x+x_ref,y_utm=y+y_ref).values
        verts = np.array([[x-100, y-100, z],
                          [x+100, y-100, z],
                          [x+100, y+100, z],
                          [x-100, y+100, z],
                          [x-100, y-100, z]  # close the loop
                                            ])
        ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], color='b', linewidth=2, zorder=10)
        
for x,y,D,H in zip(Turbines.x_utm.values-x_ref,
                   Turbines.y_utm.values-y_ref,
                   Turbines.Diameter.values,
                   Turbines['Hub height'].values):
    if np.isnan(D):
        D=127
    if np.isnan(H):
        H=90
    if x>xlim[0] and x<xlim[1] and y>ylim[0] and y<ylim[1]:
        z=Topo.elevation.interp(x_utm=x+x_ref,y_utm=y+y_ref).values
        plt.plot([x,x],[y,y],[z,z+H],color=(0.5,0.5,0.5,1),linewidth=2, zorder=10)
        plt.plot([x,x+D/2*np.cos(np.radians(30))*ar],[y,y],[z+H,z+H-D/2*np.sin(np.radians(30))],color=(0.5,0.5,0.5,1), zorder=10)
        plt.plot([x,x-D/2*np.cos(np.radians(30))*ar],[y,y],[z+H,z+H-D/2*np.sin(np.radians(30))],color=(0.5,0.5,0.5,1), zorder=10)
        plt.plot([x,x],[y,y],[z+H,z+H+D/2],color=(0.5,0.5,0.5,1), zorder=10)
                
ax.set_aspect('equal')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_box_aspect(ax.get_box_aspect()*np.array([1,1,ar]))
plt.tight_layout()

ax.xaxis.pane.set_visible(False)
ax.yaxis.pane.set_visible(False)
ax.zaxis.pane.set_visible(False)

# Hide grid lines
ax.xaxis._axinfo["grid"]["linewidth"] = 0
ax.yaxis._axinfo["grid"]["linewidth"] = 0
ax.zaxis._axinfo["grid"]["linewidth"] = 0
ax.set_axis_off()

#turbulent wake
fig=plt.figure(figsize=(18,8))
ax = fig.add_subplot(111, projection='3d')
utl.draw_turbine_3d(ax, 0, 0, 0, 1, 0.7,- 90)
epsilon_wake=0.25*(0.5*(1+(1-ct)**0.5)/(1-ct)**0.5)**0.5

x=np.arange(0,wake_lenght,0.01)
y=np.arange(-1,1.1,0.025)
z=np.arange(-1,1.1,0.025)
yc=np.sin(np.radians(x*200))*0.25
wake=xr.Dataset()
wake['yc']=xr.DataArray(yc,coords={'x':x})
wake['DU']=xr.DataArray(np.zeros((len(x),len(y),len(z))),coords={'x':x,'y':y,'z':z})

wake['DU']=(1-(1-ct/(8*(k_wake*wake.x+epsilon_wake)**2)))*np.exp(-1/(2*(k_wake*wake.x+epsilon_wake)**2)*((wake.y+wake.yc)**2+wake.z**2))
ctr=0
for w in wake_iso:
    sel=np.where(np.abs(wake.DU.values.ravel()-w)<0.01)[0]
    X=wake.x.expand_dims({'y': y,'z':z}).transpose('x', 'y','z').values
    Y=wake.y.expand_dims({'x': x,'z':z}).transpose('x', 'y','z').values
    Z=wake.z.expand_dims({'x': x,'y':y}).transpose('x', 'y','z').values
    plt.plot(X.ravel()[sel],Y.ravel()[sel],Z.ravel()[sel],'.',color=colors_wake[ctr],markersize=1,alpha=0.5)
    ctr+=1
                
ax.set_aspect('equal')
    
ax.set_axis_off()


#turbulent wake
fig=plt.figure(figsize=(18,8))
ax = fig.add_subplot(111, projection='3d')
utl.draw_turbine_3d(ax, 0, 0, 0, 1, 0.7,-130)
epsilon_wake=0.25*(0.5*(1+(1-ct)**0.5)/(1-ct)**0.5)**0.5

x=np.arange(0,wake_lenght,0.01)
y=np.arange(-2,1.1,0.025)
z=np.arange(-1,1.1,0.025)
yc=np.sin(np.radians(x*30))
wake=xr.Dataset()
wake['yc']=xr.DataArray(yc,coords={'x':x})
wake['DU']=xr.DataArray(np.zeros((len(x),len(y),len(z))),coords={'x':x,'y':y,'z':z})

wake['DU']=(1-(1-ct/(8*(k_wake*wake.x+epsilon_wake)**2)))*np.exp(-1/(2*(k_wake*wake.x+epsilon_wake)**2)*((wake.y+wake.yc)**2+wake.z**2))
ctr=0
for w in wake_iso:
    sel=np.where(np.abs(wake.DU.values.ravel()-w)<0.01)[0]
    X=wake.x.expand_dims({'y': y,'z':z}).transpose('x', 'y','z').values
    Y=wake.y.expand_dims({'x': x,'z':z}).transpose('x', 'y','z').values
    Z=wake.z.expand_dims({'x': x,'y':y}).transpose('x', 'y','z').values
    plt.plot(X.ravel()[sel],Y.ravel()[sel],Z.ravel()[sel],'.',color=colors_wake[ctr],markersize=0.5,alpha=0.5)
    ctr+=1
                
utl.draw_turbine_3d(ax, 3, 0, 0, 1, 0.7,-90)

ax.set_aspect('equal')
    
ax.set_axis_off()

plt.figure()
plt.contourf(X_terrain,Z_terrain,ws_terrain,np.arange(0,15.1,0.25),cmap='coolwarm')
plt.contour(X_terrain,Z_terrain,ws_terrain,np.arange(0,15.1,0.25),colors='k',linewidhts=0.1,alpha=0.25)
plt.plot(r_terrain,z_terrain,'k')
plt.gca().set_facecolor('g')

