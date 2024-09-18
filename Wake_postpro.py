# -*- coding: utf-8 -*-
"""
Postprocessing of wake stats
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
import glob
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
from scipy import stats
import pandas as pd

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
source_config=os.path.join(cd,'config.yaml')
source_log='data/20230101.000500-20240101.224500.awaken.glob.summary.csv'
source_stats=os.path.join(cd,'data/awaken/rt1.lidar.z02.a0/unwaked/*stats.nc')

#site
D=127#[m] rotor diameter
H=90#[m] hub height

#stats
max_TI=50#[%] maximum TI
max_err_u=0.2#maximum error in mean normalized streamwise velocity
max_err_TI=4#maximum error in mean turbulence intensity
bins_WS=np.arange(0,15.1,0.25)
bins_TI=np.arange(0,31,0.5)

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

#log
LOG=pd.read_csv(os.path.join(cd,source_log)).replace(-9999, np.nan)
LOG['Rotor-averaged TI [%]']=LOG['Rotor-averaged TI [%]'].where(LOG['Rotor-averaged TI [%]']>0).where(LOG['Rotor-averaged TI [%]']<max_TI)

#get filenames
files=sorted(glob.glob(source_stats))

#zeroing
WS_bin_all=[]
WD_bin_all=[]
TI_bin_all=[]
N_all=[]

#graphics 
fig_xz_U,axs_xz_U = plt.subplots(len(files), 1, figsize=(18, 10))
fig_xz_TI,axs_xz_TI = plt.subplots(len(files), 1, figsize=(18, 10))

#%% Main
ctr=1
for s in files:
    
    #read stats
    Data=xr.open_dataset(s)
    x=Data.x.values
    y=Data.y.values
    z=Data.z.values
    Y,Z=np.meshgrid(y,z,indexing='ij')
    X1,Z1=np.meshgrid(x,z,indexing='ij')
    
    #qc
    Data['u_avg_qc']= Data['u_avg'].where(Data['u_top']-Data['u_low']<max_err_u)
    Data['TI_avg_qc']= Data['TI_avg'].where(Data['TI_top']-Data['TI_low']<max_err_TI)
    
    WS_bin_all.append(Data.attrs['WS_bin'])
    WD_bin_all.append(Data.attrs['WD_bin'])
    TI_bin_all.append(Data.attrs['TI_bin'])
    N_all.append(len(Data['files']))
    
    #plot 3D wake (U)
    fig = plt.figure(figsize=(18,10))
    ax = fig.add_subplot(111, projection='3d')
    for xp in x_plot:
        i=np.where(xp==x/D)[0][0]
        cf=ax.contourf(Data['u_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(0.5,1.5,0.05),extend='both',cmap='coolwarm')
        ax.contour(Data['u_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(0.5,1.5,0.05),colors='k',linewidths=0.5,linestyles='solid')
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
    plt.title(s,fontsize=12)
    
    utl.draw_turbine_3d(ax,0,0,H/D,1,H/D,90)
    cb=plt.colorbar(cf,label=r'$\overline{u} ~ U_\infty^{-1}$')
    cb.ax.set_position([0.8, 0.2, 0.015, 0.6]) 
    
    plt.savefig(s.replace('.nc','.U.png'))
    plt.close(fig)
    
    #plot 3D wake (TI)
    fig = plt.figure(figsize=(18,10))
    ax = fig.add_subplot(111, projection='3d')
    for xp in x_plot:
        i=np.where(xp==x/D)[0][0]
        cf=ax.contourf(Data['TI_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(0,21),extend='both',cmap='hot')
        ax.contour(Data['TI_avg_qc'].values[i,:,:], -Y/D, Z/D+H/D,zdir='x',offset=-xp,levels=np.arange(0,21),colors='k',linewidths=0.5,linestyles='solid')
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
    plt.title(s,fontsize=12)
    
    utl.draw_turbine_3d(ax,0,0,H/D,1,H/D,90)
    cb=plt.colorbar(cf,label=r'$I_u$ [%]')
    cb.ax.set_position([0.8, 0.2, 0.015, 0.6]) 

    plt.savefig(s.replace('.nc','.TI.png'))
    plt.close(fig)
    
    #plot wake cross sections
    plt.figure(fig_xz_U)
    plt.sca(axs_xz_U[ctr-1])
    cf_U=plt.contourf(X1/D,Z1/D+H/D,Data['u_avg_qc'].sel(y=0), levels=np.arange(0.5,1.5,0.05),extend='both',cmap='coolwarm')
    plt.contour(X1/D,Z1/D+H/D,Data['u_avg_qc'].sel(y=0), levels=np.arange(0.5,1.5,0.05),colors='k',linewidths=0.5,linestyles='solid')
    plt.xlim([xmin, xmax])
    plt.ylim([zmin, zmax])
    utl.draw_turbine(0,H/D, 1,270)
    if ctr==len(files):
        plt.xlabel(r'$x/D$')
    plt.ylabel(r'$x/D$')
    if ctr==1:
        plt.title(r'$\theta_w\in['+str(Data.attrs['WD_bin'][0])+','+str(Data.attrs['WD_bin'][1])+r')^\circ$')
        
    plt.text(x=0,y=1.35,s=r'$U_\infty\in['+str(Data.attrs['WS_bin'][0])+','+str(Data.attrs['WS_bin'][1])+r')$ m s$^{-1}$, TI$_\infty \in['+str(Data.attrs['TI_bin'][0])+','+str(Data.attrs['TI_bin'][1])+')\%$',
             bbox={'alpha':0.5,'facecolor':'w'})
    plt.grid()
    
    plt.figure(fig_xz_TI)
    plt.sca(axs_xz_TI[ctr-1])
    cf_TI=plt.contourf(X1/D,Z1/D+H/D,Data['TI_avg_qc'].sel(y=0), levels=np.arange(0,21),extend='both',cmap='hot')
    plt.contour(X1/D,Z1/D+H/D,Data['TI_avg_qc'].sel(y=0), levels=np.arange(0,21),colors='k',linewidths=0.5,linestyles='solid')
    plt.xlim([xmin, xmax])
    plt.ylim([zmin, zmax])
    utl.draw_turbine(0,H/D, 1,270)
    if ctr==len(files):
        plt.xlabel(r'$x/D$')
    plt.ylabel(r'$x/D$')
    if ctr==1:
        plt.title(r'$\theta_w\in['+str(Data.attrs['WD_bin'][0])+','+str(Data.attrs['WD_bin'][1])+r')^\circ$')
        
    plt.text(x=0,y=1.35,s=r'$U_\infty\in['+str(Data.attrs['WS_bin'][0])+','+str(Data.attrs['WS_bin'][1])+r')$ m s$^{-1}$, TI$_\infty \in['+str(Data.attrs['TI_bin'][0])+','+str(Data.attrs['TI_bin'][1])+')\%$',
             bbox={'alpha':0.5,'facecolor':'w'})
    plt.grid()

    ctr+=1

#WS-TI histogram
if np.max(np.std(np.array(WD_bin_all),axis=0))>0:
    raise BaseException('Different wind sectors')
WD_bin=WD_bin_all[-1]

WS=LOG['Hub-height wind speed [m/s]'].values
WD=LOG['Hub-height wind direction [degrees]'].values
TI=LOG['Rotor-averaged TI [%]'].values

real=~np.isnan(WS+WD+TI)
if WD_bin[0]<WD_bin[1]:
    sel_WD=(WD>=WD_bin[0])*(WD<WD_bin[1])
else:
    sel_WD=(WD>=WD_bin[0])*(WD<360)+(WD>=0)*(WD<WD_bin[1])

N=stats.binned_statistic_2d(WS[sel_WD*real],TI[sel_WD*real],WS[sel_WD*real],statistic='count',bins=[bins_WS,bins_TI])[0]

#%% Plots

#finalize plots
plt.figure(fig_xz_U)
cb=fig_xz_U.colorbar(cf_U, ax=axs_xz_U, orientation='vertical', fraction=0.02, pad=0.04,label=r'$\overline{u} ~ U_\infty^{-1}$')
cb.ax.set_position([0.92, 0.1, 0.03, 0.83])  
plt.tight_layout(rect=[0, 0, 0.9, 1])  
plt.figure(fig_xz_TI)
cb=fig_xz_TI.colorbar(cf_TI, ax=axs_xz_TI, orientation='vertical', fraction=0.02, pad=0.04,label=r'$I_u$ [%]')
cb.ax.set_position([0.92, 0.1, 0.03, 0.83])  
plt.tight_layout(rect=[0, 0, 0.9, 1])  

#bin density
plt.figure()
plt.pcolor(utl.mid(bins_WS),utl.mid(bins_TI),np.log10(N.T),vmin=0,vmax=np.log10(50),cmap='inferno')
for ws,ti,n in zip(WS_bin_all,TI_bin_all,N_all):
    plt.plot([ws[0],ws[0],ws[1],ws[1]],[ti[0],ti[1],ti[1],ti[0]],'--g',linewidth=4)
    plt.text(ws[0]+(ws[1]-ws[0])/2-0.5,ti[0]+(ti[1]-ti[0])/2-1,str(n),color='g',bbox={'alpha':0.5,'facecolor':'w'})
plt.gca().set_facecolor((0,0,0,0.25))
plt.xlabel(r'$U_\infty$ [m s$^{-1}$]')
plt.ylabel(r'TI$_\infty$ [%]')
plt.xticks(np.arange(0,16))
plt.yticks(np.arange(0,31,2.5))
plt.grid()
cb=plt.colorbar(label='Occurence')
cb.set_ticks(np.log10(np.array([1,2,5,10,20,50])))
cb.set_ticklabels([1,2,5,10,20,50])
plt.title(r'$\theta_w\in['+str(WD_bin[0])+','+str(WD_bin[1])+r')^\circ$')
plt.tight_layout()


