# -*- coding: utf-8 -*-
"""
RHI processing
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['savefig.dpi'] = 200
plt.close('all')

#%% Inputs

source=os.path.join(cd,'data/awaken/rt1.lidar.z02.b0/rt1.lidar.z02.b0.20231130.092004.user5.rhi.nc')
wd_ref=180#[deg] aligned wind directipon
min_cos=1/3 #minimum cosine for de-projection
z_sel=500
dz=40
#%% Initialization
Data=xr.open_dataset(source)

os.makedirs(os.path.join(cd,'figures',os.path.basename(source)),exist_ok=True)

#%% Main
#tilt correction
Data['elevation']=Data.elevation+Data.pitch.median()
Data['x_corr']=Data.range*np.cos(np.radians(Data.elevation))*np.cos(np.radians(90-Data.azimuth))
Data['z_corr']=Data.range*np.sin(np.radians(Data.elevation))
                
Data=Data.where(Data.qc_wind_speed==0).where(np.abs(np.cos(np.radians(Data.elevation)))>min_cos)
real=~np.isnan(Data.x_corr+Data.z_corr+Data.wind_speed).values

u_eq=-Data.wind_speed/np.cos(np.radians(Data.elevation))

rws_sel=Data.wind_speed.where(Data.z>z_sel-dz/2).where(Data.z<z_sel+dz/2)
x_sel=Data.x.where(Data.z>z_sel-dz/2).where(Data.z<z_sel+dz/2)

#%% Plots
for i in Data.scanID:
    plt.figure(figsize=(18,4))
    plt.scatter(Data.x_corr.sel(scanID=i).values,
                Data.z_corr.sel(scanID=i).values,s=1,
                c=u_eq.sel(scanID=i).values,cmap='coolwarm')
    
    plt.title(os.path.basename(source)+' '+str(int(i)))
    ax=plt.gca()
    ax.set_aspect('equal')
    plt.xlim([-2000,8500])
    plt.ylim([0,1250])
    plt.xlabel(r'$x$ [m]')
    plt.ylabel(r'$y$ [m]')
    plt.grid()
    plt.colorbar(label='$u/U_\infty$ [m s$^{-1}$]')
    plt.savefig(os.path.join(cd,'figures',os.path.basename(source),f'{int(i):02d}.png'))
    plt.close()

plt.figure()
x_plot=np.arange(-2000,2000)
beta_plot=np.degrees(np.arccos(x_plot/(z_sel**2+x_plot**2)**0.5))
rws_plot=18*np.cos(np.radians(beta_plot-1))
rws_plot[np.cos(np.radians(beta_plot+1))>3]=np.nan
plt.plot(x_sel.values.ravel(),rws_sel.values.ravel(),'.k')
plt.plot(x_plot,rws_plot,'r')
