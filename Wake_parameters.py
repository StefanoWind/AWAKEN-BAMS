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
sources_stats={'South':os.path.join(cd,'data/awaken/rt1.lidar.z02.a0/unwaked/*stats.nc'),
               'North':os.path.join(cd,'data/awaken/rt1.lidar.z02.a0/waked/*stats.nc')}

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

linestyles={'South':'-','North':'--'}
colors={0:'b',5:'g',10:'r'}

#%% Initialization

#log
LOG=pd.read_csv(os.path.join(cd,source_log)).replace(-9999, np.nan)
LOG['Rotor-averaged TI [%]']=LOG['Rotor-averaged TI [%]'].where(LOG['Rotor-averaged TI [%]']>0).where(LOG['Rotor-averaged TI [%]']<max_TI)

#get filenames

#zeroing

# WS_bin_all={}
# WD_bin_all={}
TI_bin_all={}
u_rot={}
TI_rot={}

#%% Main

for s in sources_stats:
    files=sorted(glob.glob(sources_stats[s]))
    u_rot[s]=[]
    TI_rot[s]=[]
    TI_bin_all[s]=[]

    for f in files:

        #read stats
        Data=xr.open_dataset(f)
        
        
        #qc
        Data['u_avg_qc']= Data['u_avg'].where(Data['u_top']-Data['u_low']<max_err_u)
        Data['TI_avg_qc']= Data['TI_avg'].where(Data['TI_top']-Data['TI_low']<max_err_TI)
        
        TI_bin_all[s].append(Data.attrs['TI_bin'])
       
        u_wake=  Data['u_avg_qc'].where((Data['y']**2+Data['z']**2)**0.5/D<0.5)
        TI_wake=Data['TI_avg_qc'].where((Data['y']**2+Data['z']**2)**0.5/D<0.5)
        
        u_rot[s]=utl.vstack(u_rot[s],u_wake.mean(dim='y').mean(dim='z').values)
        TI_rot[s]=utl.vstack(TI_rot[s],TI_wake.mean(dim='y').mean(dim='z').values)
        
#%% Plots
plt.figure(figsize=(18,8))
x=Data.x.values/D
plt.subplot(1,2,1)
for s in sources_stats:
    for i in range(len(u_rot[s])):
        plt.plot(x,u_rot[s][i,:],color=colors[TI_bin_all[s][i][0]],linestyle=linestyles[s],label=r'TI$_\infty \in['+str(TI_bin_all[s][i][0])+','+str(TI_bin_all[s][i][1])+')\%$')
plt.xlabel(r'$x/D$')
plt.ylabel(r' Rotor-averaged $\overline{u} ~ U_\infty^{-1}$')   
plt.ylim([0.5,1]) 
plt.grid()    
for s in sources_stats:
    plt.plot([100,100],'k',linestyle=linestyles[s],label=s)
plt.legend()
plt.subplot(1,2,2)
for s in sources_stats:
    for i in range(len(u_rot[s])):
        plt.plot(x,TI_rot[s][i,:],color=colors[TI_bin_all[s][i][0]],linestyle=linestyles[s])
plt.xlabel(r'$x/D$')
plt.ylabel(r' Rotor-averaged $I_u$ [%]')    
plt.grid()         


   