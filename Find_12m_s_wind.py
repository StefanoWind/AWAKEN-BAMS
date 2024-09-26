# -*- coding: utf-8 -*-
"""
Extract 12 m/s wind
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

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
WS=12#[m/s] selected wind speed

source='data/inflow_avg_S.nc'

#%% Initialization
Data=xr.open_dataset(source)
z_WS=[]

#%% Main
for h in Data.hour:
    signal=Data.WS.sel(hour=h).values-WS
    real=~np.isnan(signal)
    z_real=Data.height.values[real]
    signal_real=signal[real]
    if np.max(signal_real)*np.min(signal_real)<0:
        zc=np.where(np.diff(np.sign(signal_real)))[0][0]
        z_WS=np.append(z_WS,(z_real[zc]+z_real[zc+1])/2)
    else:
        z_WS=np.append(z_WS,np.nan)

#%% Plots
plt.figure(figsize=(18,5))
cf=plt.contourf(Data.hour,Data.height,Data.WS.T,np.arange(0,15.5,0.5),cmap='coolwarm',extend='both')
plt.contour(Data.hour,Data.height,Data.WS.T,np.arange(0,15.5,0.5),colors='k',linewidths=0.5,linestyles='solid')
plt.plot(Data.hour,z_WS,'.g')
for z,h in zip(z_WS,Data.hour):
    if ~np.isnan(z):
        plt.text(h-0.25,z+20,s='$'+str(int(z))+'$ m',fontsize=12,color='g')
plt.xlabel('Hour (UTC)')
plt.ylabel(r'$z$ [m AGL]')
plt.grid()
plt.title(source)
plt.colobar(label=r'$U_\infty$ [m s$^{-1}$]')
plt.tight_layout()
