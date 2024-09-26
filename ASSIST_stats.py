# -*- coding: utf-8 -*-
"""
Calculated psd of inflow variables 
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
import yaml
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
from scipy import stats
import matplotlib.gridspec as gridspec

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source_config=os.path.join(cd,'config.yaml')
source_log=os.path.join(cd,'data/20230101.000500-20240101.224500.awaken.glob.summary.csv')

#dataset
source=os.path.join(cd,'data/sb.assist.z01.c0.20230101.20240101.41_levels.nc')

#qc
max_TKE=10#[m^2/s^2] maximum TKE
min_lwp=10#[g/m^2] for liquid water paths lower than this, remove clouds
max_gamma=5#maximum weight of the prior
max_rmsr=5#maximum rms of the retrieval

#stats
bin_month=np.arange(0,12.1,3)
bin_hour=np.arange(25)
bin_wd=[315,45,135,225,315]#[deg]
perc_lim=[5,95]#[%] outlier rejection
p_value=0.05
max_err={'temperature':4,'waterVapor':2,'theta':2}
max_gap=3600#[s] maximum data gap

#user
variables=['temperature','waterVapor']

#graphics
wd_name={315:'N',45:'E',135:'S',225:'W'}
clim={'temperature':[5,30],'waterVapor':[0,10]}
labels={'temperature':r'$T$ [$^\circ$C]','waterVapor':r'r [g Kg$^{-1}$]'}
ticks={'temperature':np.arange(5,31),'waterVapor':np.arange(0,11)}
# ticklabels={'temperature':np.arange(5,31),'waterVapor':np.round(10**np.arange(-1.5,0.5,0.1),2)}

#%% Initialization
AST=xr.open_dataset(source)
LOG=pd.read_csv(source_log)

#inflow
tnum_log=np.array([utl.datenum(t,'%Y-%m-%d %H:%M:%S') for t in LOG['UTC Time']])
time_log=np.array([utl.num_to_dt64(t) for t in tnum_log])
wd=LOG['Hub-height wind direction [degrees]'].values
wd[wd==-9999]=np.nan

#inflow
tnum_log=np.array([utl.datenum(t,'%Y-%m-%d %H:%M:%S') for t in LOG['UTC Time']])
time_log=np.array([utl.num_to_dt64(t) for t in tnum_log])
wd=LOG['Hub-height wind direction [degrees]'].values
wd[wd==-9999]=np.nan

#time stats
month=np.array([int(str(t)[5:7]) for t in time_log])
month[month==12]=0#december to 0
hour=np.mod(tnum_log,3600*24)/3600#time of the day
hour_avg=utl.mid(bin_hour)

#qc
AST['height']=AST.height*1000
AST['cbh']=AST.cbh*1000
AST['cbh'][AST['lwp']<min_lwp]=10000

for v in variables: 
    AST[v]=AST[v].where(AST['height']<AST['cbh']).where(AST['rmsr']<max_rmsr).where(AST['gamma']<max_gamma)
    
AST_avg={}
AST_int=xr.Dataset()


#%% Main

height=AST.height.values
tnum=np.float64(AST.time)/10**9
AST['timestamp']=xr.DataArray(data=tnum,coords={'time':AST.time}).expand_dims({'height':AST.height})

for v in variables:    
    tnum_real=AST['timestamp'].where(~np.isnan(AST[v]))
    diff_time=tnum_real.interp(time=time_log,method='nearest')-xr.DataArray(data=tnum_log,coords={'time':time_log})
    AST_int[v]=AST[v].interp(time=time_log).where(np.abs(diff_time)<max_gap)
    
    
AST_int['month']=xr.DataArray(data=month,coords={'time':time_log})
AST_int['WD']=    xr.DataArray(data=wd,coords={'time':time_log})

#daily average (wind sector)
for wd1,wd2 in zip(bin_wd[:-1],bin_wd[1:]):
    if wd1>wd2:
        AST_sel=AST_int.where(((AST_int['WD']>=wd1) & (AST_int['WD']<360)) | ((AST_int['WD']>=0) & (AST_int['WD']<wd2)))
    else:
        AST_sel=AST_int.where(AST_int['WD']>=wd1).where(AST_int['WD']<wd2)
    AST_avg[wd_name[wd1]]=xr.Dataset()
    for v in variables:
        f_avg_all=[]
        for h in height:
            print(v+': '+str(h))
            f=AST_sel[v].sel(height=h).values
            f[f==0]=np.nan
            real=~np.isnan(f)
            
            if np.sum(real)>0:
                f_avg= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_mean(x,perc_lim=perc_lim),                             bins=bin_hour)[0]
                f_low= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_BS_mean(x,perc_lim=perc_lim,p_value=p_value/2*100),    bins=bin_hour)[0]
                f_top= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_BS_mean(x,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=bin_hour)[0]
    
                f_avg[np.abs(f_top-f_low)>max_err[v]]=np.nan
                f_avg[np.isnan(np.abs(f_top-f_low))]=np.nan
                
                f_avg_all=utl.vstack(f_avg_all,f_avg)

            else:
                f_avg_all=utl.vstack(f_avg_all,hour_avg*np.nan)
           
        AST_avg[wd_name[wd1]][v]=xr.DataArray(data=f_avg_all.T,coords={'hour':hour_avg,'height':AST_sel.height.values}).interpolate_na(dim='height',limit=2).interpolate_na(dim='hour',limit=1)

#%% Plots
#wind sectors
fig=plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(len(wd_name)+1, len(variables), height_ratios=[0.2]+[1]*len(wd_name))  # 0.1 row for the colorbars

ctr1=0
for v in variables:
    ctr2=0
    for w in wd_name:
        wdn=wd_name[w]
        ax=fig.add_subplot(gs[ctr2+1,ctr1])
        cf=plt.contourf(hour_avg,AST_avg[wdn].height,AST_avg[wdn][v].T,ticks[v],cmap='coolwarm',extend='both')
        plt.contour(hour_avg,AST_avg[wdn].height,AST_avg[wdn][v].T,ticks[v],colors='k',linewidths=0.5,linestyles='solid')
        if ctr1==0:
            plt.ylabel(r'$z$ [m AGL]')
            plt.text(0.3,1500,wdn,fontweight='bold',bbox={'alpha':0.25,'facecolor':'w'})
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0,6,12,18,24])
        if ctr2==len(wd_name)-1:
            plt.xlabel('Hour (UTC)')
        else:
            ax.set_xticklabels([])
            
        plt.grid()
        ctr2+=1
        
        
    cbar_ax = ax=fig.add_subplot(gs[0,ctr1])
    cbar=fig.colorbar(cf, cax=cbar_ax, orientation='horizontal',label=labels[v])
    cbar.set_ticks(ticks[v])
    # cbar.set_ticklabels(ticklabels[v])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_tick_params(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
        
    ctr1+=1
plt.tight_layout()