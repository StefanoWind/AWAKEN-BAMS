# -*- coding: utf-8 -*-
"""
Calculated psd of inflow variables
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
source_config=os.path.join(cd,'config.yaml')
source_log='data/20230101.000500-20240101.224500.awaken.sa1.summary.csv'
source_lidar='data/awaken/sa1.lidar.z03.c1/*nc'

#stats
WD_range=[100,260]#[deg] wind direction range
bin_hour=np.arange(25)
perc_lim=[5,95]
p_value=0.95
M_BS=100

#user-defined
variables=['WS','WD','TKE']
max_err={'WS':1,'WD':10,'TKE':0.25}

# graphics
barb_stagger_height=10

#%% Initialization

#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl


#load log
IN=pd.read_csv(os.path.join(cd,source_log)).replace(-9999, np.nan)

#%lidar data
LID=xr.open_mfdataset(glob.glob(os.path.join(cd,source_lidar)))

hour_avg=utl.mid(bin_hour)

LID_avg=xr.Dataset()

#%% Main 

#select wind direction range
tnum_in=np.array([utl.datenum(t,'%Y-%m-%d %H:%M:%S') for t in IN['UTC Time']])
tnum_lid=np.array([utl.dt64_to_num(t) for t in LID.time.values])  
WD_int=np.interp(tnum_lid,tnum_in,IN['Hub-height wind direction [degrees]'].values)
LID['WD_hh']=xr.DataArray(data=WD_int,coords={'time':LID.time})
LID_sel=LID.where(LID['WD_hh']>WD_range[0]).where(LID['WD_hh']<WD_range[-1])

#daily average
hour=(tnum_lid-utl.floor(tnum_lid,3600*24))/3600
height=LID_sel.height.values

for v in variables:
    f_avg_all=[]
    for h in height:
        print(h)
        f=LID_sel[v].sel(height=h).values
        real=~np.isnan(f)
        
        if np.sum(real)>0:
            
            if v=='WD':
                f_avg= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_mean(x,perc_lim),                          bins=bin_hour)[0]
                f_low= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_BS_mean(x,perc_lim,p_value/2*100,M_BS),    bins=bin_hour)[0]
                f_top= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_BS_mean(x,perc_lim,(1-p_value/2)*100,M_BS),bins=bin_hour)[0]
            else:
                c=utl.cosd(f)
                c_avg= stats.binned_statistic(hour[real], c[real],statistic=lambda x:utl.filt_mean(x,perc_lim),                          bins=bin_hour)[0]
                c_low= stats.binned_statistic(hour[real], c[real],statistic=lambda x:utl.filt_BS_mean(x,perc_lim,p_value/2*100,M_BS),    bins=bin_hour)[0]
                c_top= stats.binned_statistic(hour[real], c[real],statistic=lambda x:utl.filt_BS_mean(x,perc_lim,(1-p_value/2)*100,M_BS),bins=bin_hour)[0]
                
                s=utl.sind(f)
                s_avg= stats.binned_statistic(hour[real], s[real],statistic=lambda x:utl.filt_mean(x,perc_lim),                         bins=bin_hour)[0]
                s_low= stats.binned_statistic(hour[real], s[real],statistic=lambda x:utl.filt_BS_mean(x,perc_lim,p_value/2*100,M_BS),    bins=bin_hour)[0]
                s_top= stats.binned_statistic(hour[real], s[real],statistic=lambda x:utl.filt_BS_mean(x,perc_lim,(1-p_value/2)*100,M_BS),bins=bin_hour)[0]
    
                f_avg=utl.cart2pol(c_avg,s_avg)[1]
                f_low=utl.cart2pol(c_low,s_low)[1]
                f_top=utl.cart2pol(c_top,s_top)[1]
    
            f_avg[np.abs(f_top-f_low)>max_err[v]]=np.nan
            
            f_avg_all=utl.vstack(f_avg_all,f_avg)

        else:
            f_avg_all=utl.vstack(f_avg_all,hour_avg*np.nan)
       
    LID_avg[v]=xr.DataArray(data=f_avg_all.T,coords={'hour':hour_avg,'height':LID_sel.height.values})
    
LID_avg['U']=LID['WS']*utl.cosd(270-LID['WD'])
LID_avg['V']=LID['WS']*utl.sind(270-LID['WD'])

#%% Plots
fig=plt.figure(figsize=(18,10))
ax=plt.subplot(2,1,1)
CS = ax.contourf(hour_avg, height, LID_avg['WS'].T, np.arange(3, 12,0.5), extend='both', cmap='viridis')
ax.barbs(hour_avg, height[::barb_stagger_height], LID_avg['U'][::barb_stagger_height,:]*1.94, LID_avg['V'][::barb_stagger_height,:]*1.94,
barbcolor='k', flagcolor='k', color='k', fill_empty=0, length=5.8, linewidth=1)
ax.barbs(hour_avg[0]+np.timedelta64(1260,'s'), 2600, 10*np.cos(60), -10*np.sin(60), barbcolor='k',
flagcolor='k', color='k', fill_empty=0, length=5.8, linewidth=1.4)
ax.text(hour_avg[0]+np.timedelta64(600,'s'), 2480, '10 kts \n', fontsize=12, bbox=dict(facecolor='none', edgecolor='black', alpha=0.8))

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size = '2%', pad=0.65)
cb = fig.colorbar(CS, cax=cax, orientation='vertical')
cb.set_label(r'Mean horizontal wind speed [m s$^{-1}$]')

ax.set_ylabel(r'z [m $AGL$]')
ax.set_ylim(0, 3000)
ax.grid()
ax.tick_params(axis='both', which='major')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H-%M'))

ax=plt.subplot(2,1,2)
CS = ax.contourf(hour_avg, height, np.log10(LID_avg['TKE'].T), np.arange(-2,0.71,0.1),extend='both', cmap='inferno')
ax.barbs(hour_avg, height[::barb_stagger_height], LID_avg['U'][::barb_stagger_height,:]*1.94, LID_avg['V'][::barb_stagger_height,:]*1.94,
barbcolor='k', flagcolor='k', color='k', fill_empty=0, length=5.8, linewidth=1)

cb.set_label(r'Turbulent kinetic energy [m$^2$ s$^{-2}$]')
cb.set_ticks([-2,-1,0,np.log10(5)])
cb.set_ticklabels(['0.01','0.1','1','5'])

ax.set_xlabel('Time (UTC)')
ax.set_ylabel(r'z [m $AGL$]')
ax.set_ylim(0, 3000)
ax.grid()
ax.tick_params(axis='both', which='major')
