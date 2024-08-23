# -*- coding: utf-8 -*-

"""
Postporcessing of inflow psd
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
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18


#%% Inputs
source=os.path.join(cd,'data/100-260.psd.full.nc')
bin_h=np.array([100,150,300,500,1000,2000])

#%% Initialization
Data=xr.open_dataset(source)

Data_avg=Data.groupby_bins('height',bins=bin_h).mean()

colormap = plt.cm.viridis
colors = [colormap(i) for i in np.linspace(0,1,len(bin_h)-1)]

Data_avg = Data_avg.interpolate_na(dim="period", method="linear")

#%% Plots
plt.close('all')

#line plot
ctr=1
plt.figure(figsize=(18,6))
for v in Data.var():
    ax=plt.subplot(1,len(Data.var()),ctr)
    ctr2=0
    for h in Data_avg.height_bins.values:
        plt.loglog(Data_avg.period/3600,Data_avg[v].sel(height_bins=h),label=f'${h.left}<z\leq{h.right}$ m AGL',color=colors[ctr2])
        ctr2+=1
    plt.xticks([2,6,12,24,48,72],labels=['2','6','12','24','48','72'])
    plt.ylim([10**-9,10**-4])
    plt.xlabel('Period [hours]')
    if ctr==0:
        plt.ylabel('Normalized PSD [hours$^{-1}$]')
    
    plt.title(v)
    plt.grid()
    
    ctr+=1
    
        
plt.legend(draggable=True)

#heatmap
ctr=1
plt.figure(figsize=(18,6))
for v in Data.var():
    ax=plt.subplot(1,len(Data.var()),ctr)
    plt.contourf(Data.period/3600,Data.height,np.log10(Data[v].T),np.arange(-9,-4,0.1),cmap='viridis',extent='both')
    plt.xlabel('Period [hours]')
    plt.ylabel(r'$z$ [m AGL]')
    plt.xlim([2,48])
    plt.ylim([0,2000])
    ax.set_xscale('log')
    plt.title(v)
    plt.grid()
    ctr+=1