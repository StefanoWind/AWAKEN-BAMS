# -*- coding: utf-8 -*-
"""
Compare different wake stats
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

#%% Inputs
source_config=os.path.join(cd,'config.yaml')
source_log='data/20230101.000500-20240101.224500.awaken.glob.summary.csv'
source1=os.path.join(cd,'data/awaken/rt1.lidar.z02.a0/unwaked/5.8.135.225.00.05.stats.nc')
source2=os.path.join(cd,'data/awaken/rt1.lidar.z02.a0/waked/5.8.0.90.00.05.stats.nc')

#site
D=127#[m] rotor diameter
H=90#[m] hub height

#stats
max_TI=50#[%] maximum TI
max_err_u=0.2#maximum error in mean normalized streamwise velocity
max_err_TI=4#maximum error in mean turbulence intensity

#graphics
#plot area[D]
xmin=-0.1
xmax=10
ymin=-1.5
ymax=1.5
zmin=0
zmax=200/D

x_plot=np.array([2,3,4.5,6,8,10])#[D] plot locations

#%% Initialization

#load data
Data1=xr.open_dataset(source1)
x=Data1.x.values
y=Data1.y.values
z=Data1.z.values
Y,Z=np.meshgrid(y,z,indexing='ij')

Data2=xr.open_dataset(source2)

#%% Main
#qc
Data1['u_avg_qc']= Data1['u_avg'].where(Data1['u_top']-Data1['u_low']<max_err_u)
Data1['TI_avg_qc']= Data1['TI_avg'].where(Data1['TI_top']-Data1['TI_low']<max_err_TI)
Data2['u_avg_qc']= Data2['u_avg'].where(Data2['u_top']-Data2['u_low']<max_err_u)
Data2['TI_avg_qc']= Data2['TI_avg'].where(Data2['TI_top']-Data2['TI_low']<max_err_TI)

#%% Plots

#plot 3D wake (U1)
fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111, projection='3d')
for xp in x_plot:
    i=np.where(xp==x/D)[0][0]
    cf=ax.contourf(Data1['u_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(0.5,1.5,0.05),extend='both',cmap='coolwarm')
    ax.contour(    Data1['u_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(0.5,1.5,0.05),colors='k',linewidths=0.5,linestyles='solid')
ax.set_xlim([-xmax,-xmin])
ax.set_ylim([-ymax,-ymin])
ax.set_zlim([zmin, zmax])
ax.set_xticks(-x_plot)
ax.set_xticklabels([r'$'+str(int(xx))+'D$' for xx in x_plot])
ax.set_yticks([-1,-1/2,0,1/2,1])
ax.set_yticklabels([r'$D$',r'$D/2$',r'$0$',r'$-D/2$',r'$-D$'])
ax.set_zticks([H/D-1/2,H/D,H/D+1/2])
ax.set_zticklabels([r'$H-D/2$',r'$H$',r'$H+D/2$'])
utl.axis_equal()
ax.view_init(40,413)
plt.title(r'$\theta_w\in['+str(Data1.attrs['WD_bin'][0])+','+str(Data1.attrs['WD_bin'][1])+r')^\circ$')

utl.draw_turbine_3d(ax,0,0,H/D,1,H/D,90)
cb=plt.colorbar(cf,label=r'$\overline{u} ~ U_\infty^{-1}$')
cb.ax.set_position([0.8, 0.2, 0.015, 0.6]) 


#plot 3D wake (U2)
fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111, projection='3d')
for xp in x_plot:
    i=np.where(xp==x/D)[0][0]
    cf=ax.contourf(Data2['u_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(0.5,1.5,0.05),extend='both',cmap='coolwarm')
    ax.contour(    Data2['u_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(0.5,1.5,0.05),colors='k',linewidths=0.5,linestyles='solid')
ax.set_xlim([-xmax,-xmin])
ax.set_ylim([-ymax,-ymin])
ax.set_zlim([zmin, zmax])
ax.set_xticks(-x_plot)
ax.set_xticklabels([r'$'+str(int(xx))+'D$' for xx in x_plot])
ax.set_yticks([-1,-1/2,0,1/2,1])
ax.set_yticklabels([r'$D$',r'$D/2$',r'$0$',r'$-D/2$',r'$-D$'])
ax.set_zticks([H/D-1/2,H/D,H/D+1/2])
ax.set_zticklabels([r'$H-D/2$',r'$H$',r'$H+D/2$'])
utl.axis_equal()
ax.view_init(40,413)
plt.title(r'$\theta_w\in['+str(Data2.attrs['WD_bin'][0])+','+str(Data2.attrs['WD_bin'][1])+r')^\circ$')

utl.draw_turbine_3d(ax,0,0,H/D,1,H/D,90)
cb=plt.colorbar(cf,label=r'$\overline{u} ~ U_\infty^{-1}$')
cb.ax.set_position([0.8, 0.2, 0.015, 0.6]) 

#plot 3D wake (U2-U1)
fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111, projection='3d')
for xp in x_plot:
    i=np.where(xp==x/D)[0][0]
    cf=ax.contourf(Data2['u_avg_qc'].values[i,:,:]-Data1['u_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(-0.2,0.21,0.05),extend='both',cmap='seismic')
    ax.contour(    Data2['u_avg_qc'].values[i,:,:]-Data1['u_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(-0.2,0.21,0.05),colors='k',linewidths=0.5,linestyles='solid')
ax.set_xlim([-xmax,-xmin])
ax.set_ylim([-ymax,-ymin])
ax.set_zlim([zmin, zmax])
ax.set_xticks(-x_plot)
ax.set_xticklabels([r'$'+str(int(xx))+'D$' for xx in x_plot])
ax.set_yticks([-1,-1/2,0,1/2,1])
ax.set_yticklabels([r'$D$',r'$D/2$',r'$0$',r'$-D/2$',r'$-D$'])
ax.set_zticks([H/D-1/2,H/D,H/D+1/2])
ax.set_zticklabels([r'$H-D/2$',r'$H$',r'$H+D/2$'])
utl.axis_equal()
ax.view_init(40,413)
plt.title(r'Difference $\theta_w\in['+str(Data2.attrs['WD_bin'][0])+','+str(Data2.attrs['WD_bin'][1])+r')^\circ$ - $\theta_w\in['+str(Data1.attrs['WD_bin'][0])+','+str(Data1.attrs['WD_bin'][1])+r')^\circ$')

utl.draw_turbine_3d(ax,0,0,H/D,1,H/D,90)
cb=plt.colorbar(cf,label=r'$\Delta \overline{u} ~ U_\infty^{-1}$')
cb.ax.set_position([0.8, 0.2, 0.015, 0.6]) 
 
 
#plot 3D wake (TI1)
fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111, projection='3d')
for xp in x_plot:
    i=np.where(xp==x/D)[0][0]
    cf=ax.contourf(Data1['TI_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(0,21),extend='both',cmap='hot')
    ax.contour(    Data1['TI_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(0,21),colors='k',linewidths=0.5,linestyles='solid')
ax.set_xlim([-xmax,-xmin])
ax.set_ylim([-ymax,-ymin])
ax.set_zlim([zmin, zmax])
ax.set_xticks(-x_plot)
ax.set_xticklabels([r'$'+str(int(xx))+'D$' for xx in x_plot])
ax.set_yticks([-1,-1/2,0,1/2,1])
ax.set_yticklabels([r'$D$',r'$D/2$',r'$0$',r'$-D/2$',r'$-D$'])
ax.set_zticks([H/D-1/2,H/D,H/D+1/2])
ax.set_zticklabels([r'$H-D/2$',r'$H$',r'$H+D/2$'])
utl.axis_equal()
ax.view_init(40,413)
plt.title(r'$\theta_w\in['+str(Data1.attrs['WD_bin'][0])+','+str(Data1.attrs['WD_bin'][1])+r')^\circ$')

utl.draw_turbine_3d(ax,0,0,H/D,1,H/D,90)
cb=plt.colorbar(cf,label=r'$\overline{u} ~ U_\infty^{-1}$')
cb.ax.set_position([0.8, 0.2, 0.015, 0.6]) 

#plot 3D wake (TI2)
fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111, projection='3d')
for xp in x_plot:
    i=np.where(xp==x/D)[0][0]
    cf=ax.contourf(Data2['TI_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(0,21),extend='both',cmap='hot')
    ax.contour(    Data2['TI_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(0,21),colors='k',linewidths=0.5,linestyles='solid')
ax.set_xlim([-xmax,-xmin])
ax.set_ylim([-ymax,-ymin])
ax.set_zlim([zmin, zmax])
ax.set_xticks(-x_plot)
ax.set_xticklabels([r'$'+str(int(xx))+'D$' for xx in x_plot])
ax.set_yticks([-1,-1/2,0,1/2,1])
ax.set_yticklabels([r'$D$',r'$D/2$',r'$0$',r'$-D/2$',r'$-D$'])
ax.set_zticks([H/D-1/2,H/D,H/D+1/2])
ax.set_zticklabels([r'$H-D/2$',r'$H$',r'$H+D/2$'])
utl.axis_equal()
ax.view_init(40,413)
plt.title(r'$\theta_w\in['+str(Data2.attrs['WD_bin'][0])+','+str(Data2.attrs['WD_bin'][1])+r')^\circ$')

utl.draw_turbine_3d(ax,0,0,H/D,1,H/D,90)
cb=plt.colorbar(cf,label=r'$\overline{u} ~ U_\infty^{-1}$')
cb.ax.set_position([0.8, 0.2, 0.015, 0.6]) 


#plot 3D wake (TI2-TI)
fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111, projection='3d')
for xp in x_plot:
    i=np.where(xp==x/D)[0][0]
    cf=ax.contourf(Data2['TI_avg_qc'].values[i,:,:]-Data1['TI_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(-5,5.5,0.5),extend='both',cmap='seismic')
    ax.contour(    Data2['TI_avg_qc'].values[i,:,:]-Data1['TI_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(-5,5.5,0.5),colors='k',linewidths=0.5,linestyles='solid')
ax.set_xlim([-xmax,-xmin])
ax.set_ylim([-ymax,-ymin])
ax.set_zlim([zmin, zmax])
ax.set_xticks(-x_plot)
ax.set_xticklabels([r'$'+str(int(xx))+'D$' for xx in x_plot])
ax.set_yticks([-1,-1/2,0,1/2,1])
ax.set_yticklabels([r'$D$',r'$D/2$',r'$0$',r'$-D/2$',r'$-D$'])
ax.set_zticks([H/D-1/2,H/D,H/D+1/2])
ax.set_zticklabels([r'$H-D/2$',r'$H$',r'$H+D/2$'])
utl.axis_equal()
ax.view_init(40,413)
plt.title(r'Difference $\theta_w\in['+str(Data2.attrs['WD_bin'][0])+','+str(Data2.attrs['WD_bin'][1])+r')^\circ$ - $\theta_w\in['+str(Data1.attrs['WD_bin'][0])+','+str(Data1.attrs['WD_bin'][1])+r')^\circ$')

utl.draw_turbine_3d(ax,0,0,H/D,1,H/D,90)
cb=plt.colorbar(cf,label=r'$\Delta \overline{u} ~ U_\infty^{-1}$')
cb.ax.set_position([0.8, 0.2, 0.015, 0.6]) 
 
 
 
 
 
 