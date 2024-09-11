# -*- coding: utf-8 -*-
'''
Plot waked sectors for sited of interest
'''
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
source_wak=os.path.join(cd, 'data/20240910_AWAKEN_waked.nc')
source_lyt=os.path.join(cd, 'data/20231026_AWAKEN_layout.nc')
sites_plot=['A1','A2','H']

#graphics
colors={'A1':'r','A2':'b','H':'g'}

#%% Initialization
WAK=xr.open_dataset(source_wak)
wd=WAK['wind_direction'].values

TRB=xr.open_dataset(source_lyt,group='Turbines')
SIT=xr.open_dataset(source_lyt,group='Ground sites')

#site
D=TRB['Diameter'].values
D[np.isnan(D)]=np.nanmax(D)
xT=TRB['x UTM'].values
yT=TRB['y UTM'].values

#%% Main
wake_distance=WAK.sel(site=['A1','A2','H'])['waked'].values
max_wake_distance=np.nanmax(wake_distance,axis=0)
max_wake_distance_rep=np.tile(max_wake_distance[:, np.newaxis], (1, len(sites_plot))).T

unwaked=(wake_distance==max_wake_distance)+0.0
unwaked_xr=xr.DataArray(data=unwaked.T,coords={'wd':wd,'site':sites_plot})
weights=unwaked_xr/(unwaked_xr.sum(dim='site')+10**-10)
weights=weights.where(weights>0)

#%% Plots
plt.close('all')
xref=SIT['x UTM'].sel({'Site name': 'A1'}).values
yref=SIT['y UTM'].sel({'Site name': 'A1'}).values
plt.figure(figsize=(16,10))
plt.plot(xT-xref,yT-yref,'.k')
for s in sites_plot:
    x0=SIT['x UTM'].sel({'Site name': s}).values-xref
    y0=SIT['y UTM'].sel({'Site name': s}).values-yref
    plt.plot(x0,y0,'^',color=colors[s])
    plt.plot(x0+utl.cosd(90-wd)*(1/3)*3000,y0+utl.sind(90-wd)*(1/3)*3000,'--k',color='k',markersize=1)
    plt.plot(x0+utl.cosd(90-wd)*(1/2)*3000,y0+utl.sind(90-wd)*(1/2)*3000,'--k',color='k',markersize=1)
    plt.plot(x0+utl.cosd(90-wd)*(1/1)*3000,y0+utl.sind(90-wd)*(1/1)*3000,'--k',color='k',markersize=1)
    for i in range(-3,4):
        plt.plot(x0+utl.cosd(90-wd)*(i/100+weights.sel(site=s))*3000,y0+utl.sind(90-wd)*(i/100+weights.sel(site=s))*3000,'.',color=colors[s],markersize=1)

plt.xlim([-30000,+30000])
plt.ylim([-25000,+35000])
utl.axis_equal()
plt.xlabel('W-E [m]')
plt.ylabel('S-N [m]')

plt.grid()