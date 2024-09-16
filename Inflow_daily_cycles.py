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
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
from scipy import stats
import matplotlib.gridspec as gridspec

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
source_wak='data/20240910_AWAKEN_waked.nc'
sources_lid={'A1':os.path.join(cd,'data/sa1.lidar.z03.c1.20230101.20240101.41_levels.nc'),
             'A2':os.path.join(cd,'data/sa2.lidar.z01.c1.20230101.20240101.41_levels.nc'),
              'H':os.path.join(cd,'data/sh.lidar.z02.c1.20230101.20240101.41_levels.nc')}

source_ast=os.path.join(cd,'data/sb.assist.z01.c0.20230101.20240101.41_levels.nc')

source_log=os.path.join(cd,'data/20230101.000500-20240101.224500.awaken.glob.summary.csv')

#user-defined
variables_lid=['WS','TKE log']
variables_ast=[]

#stats
bin_month=np.arange(0,12.1,3)
bin_hour=np.arange(25)
bin_wd=[315,45,135,225,315]#[deg]
perc_lim=[5,95]#[%] outlier rejection
p_value=0.05
max_err={'WS':2,'TKE log':0.25,'temperature':2}#maximum bootstrap 95% confidence interval width

#qc
max_TKE=10#[m^2/s^2] maximum TKE
min_lwp=10#[g/m^2] for liquid water paths lower than this, remove clouds
max_gamma=5#maximum weight of the prior
max_rmsr=5#maximum rms of the retrieval

#graphics
month_name={0:'DJF',3:'MAM',6:'JJA',9:'SON'}
wd_name={315:'N',45:'E',135:'S',225:'W'}
clim={'WS':[5,15],'TKE log':[-1.5,0.5],'temperature':[5,30]}
labels={'WS':r'$U_\infty$ [m s$^{-1}$]','TKE log':r'TKE [m$^2$ s$^{-2}$]'}
ticks={'WS':np.arange(5,15.1,0.5),'TKE log':np.arange(-1.5,0.5,0.1)}
ticklabels={'WS':np.arange(5,15.1,0.5),'TKE log':np.round(10**np.arange(-1.5,0.5,0.1),2)}

#%% Initialization

#load data
WAK=xr.open_dataset(source_wak)
LOG=pd.read_csv(source_log)
AST=xr.open_dataset(source_ast)

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
 
#zeroing
L_D_exp=[]
LID={}
wake_distance=[]
ALL_avg={}



#%% Main
#assist qc
AST['height']=np.round(AST.height*1000)
AST['cbh']=AST.cbh*1000
AST['cbh'][AST['lwp']<min_lwp]=10000

#general coordinates
height=AST.height.values
wd_waked=WAK.wind_direction.values

#lidar data
nans=np.zeros((len(time_log),len(height),len(sources_lid)))
ctr=0
for s in sources_lid.keys():
    LID[s]=xr.open_dataset(sources_lid[s]).interp(time=time_log)
   
    #TKE qc
    TKE_qc=LID[s]['TKE'].where(LID[s]['TKE']>0).where(LID[s]['TKE']<max_TKE)
    original_nans=np.isnan(LID[s]['TKE'])
    TKE_int=TKE_qc.interpolate_na(dim='time',method='linear')
    TKE_int=TKE_int.where(original_nans==False)
    LID[s]['TKE log']=np.log10(TKE_int)
    
    #wake distance
    L_D=WAK.sel(site=s)['waked'].values#distance from nearest turbine
    L_D[np.isnan(L_D)]=9999
    
    wake_distance=utl.vstack(wake_distance,np.interp(wd,wd_waked,L_D))
    tot=0
    for v in variables_lid:
        tot+=LID[s][v]
    nans[:,:,ctr]=np.isnan(tot)
    ctr+=1

#select unwaked
max_wake_distance=np.nanmax(wake_distance,axis=0)
wake_distance_rep=    np.tile(wake_distance.T[:, np.newaxis, :],  (1, len(height), 1))
max_wake_distance_rep=np.tile(max_wake_distance[:, np.newaxis, np.newaxis], (1, len(height), len(sources_lid)))

unwaked=(wake_distance_rep==max_wake_distance_rep)+0.0
unwaked[nans==1]=0
unwaked_xr=xr.DataArray(data=unwaked,
                     coords={'time':time_log,'height':height,'site':list(sources_lid.keys())})
weights=unwaked_xr/(unwaked_xr.sum(dim='site')+10**-10)

#building unwaked lidar
ALL=xr.Dataset()
for v in variables_lid:
    ALL[v]=LID[list(sources_lid.keys())[0]][v]*0
    for s in sources_lid.keys():
        ALL[v]+=LID[s][v].fillna(-9999)*weights.sel(site=s)

#assist qc
for v in variables_ast: 
    AST[v]=AST[v].where(AST['height']<AST['cbh']).where(AST['rmsr']<max_rmsr).where(AST['gamma']<max_gamma)
      
#assist data
for v in variables_ast:    
    ALL[v]=AST[v].interp(time=time_log)
    
ALL['month']=xr.DataArray(data=month,coords={'time':time_log})
ALL['WD']=    xr.DataArray(data=wd,coords={'time':time_log})

#daily average (season)
for m1,m2 in zip(bin_month[:-1],bin_month[1:]):
    ALL_sel=ALL.where(ALL['month']>=m1).where(ALL['month']<m2)
    ALL_avg[month_name[m1]]=xr.Dataset()
    for v in variables_lid+variables_ast:
        f_avg_all=[]
        for h in height:
            print(v+': '+str(h))
            f=ALL_sel[v].sel(height=h).values
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
           
        ALL_avg[month_name[m1]][v]=xr.DataArray(data=f_avg_all.T,coords={'hour':hour_avg,'height':ALL_sel.height.values}).interpolate_na(dim='height',limit=2).interpolate_na(dim='hour',limit=1)

#daily average (wind sector)
for wd1,wd2 in zip(bin_wd[:-1],bin_wd[1:]):
    if wd1>wd2:
        ALL_sel=ALL.where(((ALL['WD']>=wd1) & (ALL['WD']<360)) | ((ALL['WD']>=0) & (ALL['WD']<wd2)))
    else:
        ALL_sel=ALL.where(ALL['WD']>=wd1).where(ALL['WD']<wd2)
    ALL_avg[wd_name[wd1]]=xr.Dataset()
    for v in variables_lid+variables_ast:
        f_avg_all=[]
        for h in height:
            print(v+': '+str(h))
            f=ALL_sel[v].sel(height=h).values
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
           
        ALL_avg[wd_name[wd1]][v]=xr.DataArray(data=f_avg_all.T,coords={'hour':hour_avg,'height':ALL_sel.height.values}).interpolate_na(dim='height',limit=2).interpolate_na(dim='hour',limit=1)

#%% Output
ALL.to_netcdf(os.path.join(cd,'data','inflow_all.nc'))
for v in ALL_avg.keys():
    ALL_avg[v].to_netcdf(os.path.join(cd,'data','inflow_avg_'+v+'.nc'))

#%% Plots

#seasons
plt.close('all')
fig=plt.figure(figsize=(16,8))
gs = gridspec.GridSpec(len(month_name)+1, len(variables_lid+variables_ast), height_ratios=[0.2]+[1]*len(month_name))  # 0.1 row for the colorbars

ctr1=0
for v in variables_lid+variables_ast:
    ctr2=0
    for m in month_name:
        mn=month_name[m]
        ax=fig.add_subplot(gs[ctr2+1,ctr1])
        cf=plt.contourf(hour_avg,ALL_avg[mn].height,ALL_avg[mn][v].T,ticks[v],cmap='coolwarm',extend='both')
        plt.contour(hour_avg,ALL_avg[mn].height,ALL_avg[mn][v].T,ticks[v],colors='k',linewidths=0.5,linestyles='solid')
        if ctr1==0:
            plt.ylabel(r'$z$ [m AGL]')
            plt.text(0.3,1500,mn,fontweight='bold',bbox={'alpha':0.25,'facecolor':'w'})
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0,6,12,18,24])
        if ctr2==len(month_name)-1:
            plt.xlabel('Hour (UTC)')
        else:
            ax.set_xticklabels([])
        plt.grid()
        ctr2+=1
        
        
    cbar_ax = ax=fig.add_subplot(gs[0,ctr1])
    cbar=fig.colorbar(cf, cax=cbar_ax, orientation='horizontal',label=labels[v])
    cbar.set_ticks(ticks[v])
    cbar.set_ticklabels(ticklabels[v])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_tick_params(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    ctr1+=1
plt.tight_layout()
    
#wind sectors
fig=plt.figure(figsize=(16,8))

ctr1=0
for v in variables_lid+variables_ast:
    ctr2=0
    for w in wd_name:
        wdn=wd_name[w]
        ax=fig.add_subplot(gs[ctr2+1,ctr1])
        cf=plt.contourf(hour_avg,ALL_avg[wdn].height,ALL_avg[wdn][v].T,ticks[v],cmap='coolwarm',extend='both')
        plt.contour(hour_avg,ALL_avg[wdn].height,ALL_avg[wdn][v].T,ticks[v],colors='k',linewidths=0.5,linestyles='solid')
        if ctr1==0:
            plt.ylabel(r'$z$ [m AGL]')
            plt.text(0.3,1500,wdn,fontweight='bold',bbox={'alpha':0.25,'facecolor':'w'})
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0,6,12,18,24])
        if ctr2==len(month_name)-1:
            plt.xlabel('Hour (UTC)')
        else:
            ax.set_xticklabels([])
            
        plt.grid()
        ctr2+=1
        
        
    cbar_ax = ax=fig.add_subplot(gs[0,ctr1])
    cbar=fig.colorbar(cf, cax=cbar_ax, orientation='horizontal',label=labels[v])
    cbar.set_ticks(ticks[v])
    cbar.set_ticklabels(ticklabels[v])
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_tick_params(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
        
    ctr1+=1
plt.tight_layout()