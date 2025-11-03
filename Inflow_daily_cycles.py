# -*- coding: utf-8 -*-
"""
Calculated psd of inflow variables
"""
import os
cd=os.path.dirname(__file__)
import utils as utl
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import stats
import matplotlib.gridspec as gridspec

mpl.rcParams.update({
"savefig.format": "png",
"savefig.dpi":500,
"pdf.fonttype": 42,
"ps.fonttype": 42,
"font.family": "serif",
"font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
"mathtext.fontset": "custom",
"mathtext.rm": "serif",
"mathtext.it": "serif:italic",
"mathtext.bf": "serif:bold",
"axes.labelsize": 16,
"axes.titlesize": 16,
"xtick.labelsize": 14,
"ytick.labelsize": 14,
"legend.fontsize": 14,
"lines.linewidth": 1,
"lines.markersize": 4,
})

#%% Inputs
source_wak='data/20240910_AWAKEN_waked.nc'#source of wake distances
sites=['A1','A2','H']#site names

#sources of lidar data
sources_lid={'A1':os.path.join(cd,'data/sa1.lidar.z03.c1.20230101.20240101.nc'),
             'A2':os.path.join(cd,'data/sa2.lidar.z01.c1.20230101.20240101.nc'),
              'H':os.path.join(cd,'data/sh.lidar.z02.c1.20230101.20240101.nc')}

source_log=os.path.join(cd,'data/glob.lidar.eventlog.avg.c2.20230101.000500.csv')#inflow table source

#user-defined
variables=['WS','TKE log'] #selected variables

#stats
bin_month=np.arange(0,12.1,3)#min in months
bin_hour=np.arange(25)#bin in hours
bin_wd=[315,45,135,225,315]#[deg] bin in wind direction
perc_lim=[5,95]#[%] outlier rejection
p_value=0.05#p-value for c.i.
max_err={'WS':2,'TKE log':0.25}#maximum bootstrap 95% confidence interval width
max_TKE=10#[m^2/s^2] maximum TKE

#graphics
month_name={0:'DJF',3:'MAM',6:'JJA',9:'SON'}
wd_name={315:'N',45:'E',135:'S',225:'W'}
clim={'WS':[5,15],'TKE log':[-1.5,0.3]}
labels={'WS':r'Wind speed [m s$^{-1}$]','TKE log':r'TKE [m$^2$ s$^{-2}$]'}
ticks={'WS':np.arange(5,15.1,0.5),'TKE log':np.arange(-1.5,0.31,0.1)}
ticklabels={'WS':np.arange(5,15.1,0.5),'TKE log':np.round(10**np.arange(-1.5,0.31,0.1),2)}

#%% Initialization

#load data
WAK=xr.open_dataset(source_wak)
LOG=pd.read_csv(source_log)

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

#general coordinates
height=xr.open_dataset(sources_lid[sites[0]]).height.values
wd_waked=WAK.wind_direction.values

#lidar data
nans=np.zeros((len(time_log),len(height),len(sources_lid)))
wake_distance=np.zeros((len(sites),len(time_log)))
ctr=0
for s in sites:
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
    
    wake_distance[ctr,:]=np.interp(wd,wd_waked,L_D)
    tot=0
    for v in variables:
        tot+=LID[s][v]
    nans[:,:,ctr]=np.isnan(tot)
    ctr+=1
    
    print(f'Data from site {s} read')

#select unwaked
max_wake_distance=np.nanmax(wake_distance,axis=0)
wake_distance_rep=    np.tile(wake_distance.T[:, np.newaxis, :],  (1, len(height), 1))
max_wake_distance_rep=np.tile(max_wake_distance[:, np.newaxis, np.newaxis], (1, len(height), len(sources_lid)))

unwaked=(wake_distance_rep==max_wake_distance_rep)+0.0
unwaked[nans==1]=0
unwaked_xr=xr.DataArray(data=unwaked,coords={'time':time_log,'height':height,'site':sites})
weights=unwaked_xr/(unwaked_xr.sum(dim='site')+10**-10)

#building unwaked lidar
ALL=xr.Dataset()
for v in variables:
    ALL[v]=LID[sites[0]][v]*0
    for s in sites:
        ALL[v]+=LID[s][v].fillna(-9999)*weights.sel(site=s)


ALL['month']=xr.DataArray(data=month,coords={'time':time_log})
ALL['WD']=xr.DataArray(data=wd,coords={'time':time_log})

#daily average
ALL_avg['tot']=xr.Dataset()
for v in variables:
    f_avg_all=[]
    for h in height:
        f=ALL[v].sel(height=h).values
        f[f==0]=np.nan
        real=~np.isnan(f)
        
        if np.sum(real)>0:
            f_avg= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_stat(x,np.nanmean,perc_lim=perc_lim),                             bins=bin_hour)[0]
            f_low= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100),    bins=bin_hour)[0]
            f_top= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=bin_hour)[0]

            f_avg[np.abs(f_top-f_low)>max_err[v]]=np.nan
            f_avg[np.isnan(np.abs(f_top-f_low))]=np.nan
            
            f_avg_all=utl.vstack(f_avg_all,f_avg)

        else:
            f_avg_all=utl.vstack(f_avg_all,hour_avg*np.nan)
    
    ALL_avg['tot'][v]=xr.DataArray(data=f_avg_all.T,coords={'hour':hour_avg,'height':ALL.height.values}).interpolate_na(dim='height',limit=2).interpolate_na(dim='hour',limit=1)


#daily average (season)
for m1,m2 in zip(bin_month[:-1],bin_month[1:]):
    ALL_sel=ALL.where(ALL['month']>=m1).where(ALL['month']<m2)
    ALL_avg[month_name[m1]]=xr.Dataset()
    for v in variables:
        f_avg_all=[]
        for h in height:
            f=ALL_sel[v].sel(height=h).values
            f[f==0]=np.nan
            real=~np.isnan(f)
            
            if np.sum(real)>0:
                f_avg= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_stat(x,np.nanmean,perc_lim=perc_lim),                             bins=bin_hour)[0]
                f_low= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100),    bins=bin_hour)[0]
                f_top= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=bin_hour)[0]
    
                f_avg[np.abs(f_top-f_low)>max_err[v]]=np.nan
                f_avg[np.isnan(np.abs(f_top-f_low))]=np.nan
                
                f_avg_all=utl.vstack(f_avg_all,f_avg)

            else:
                f_avg_all=utl.vstack(f_avg_all,hour_avg*np.nan)
            print(f'{v}: monthly avg on {m1} at {h} m done')
        ALL_avg[month_name[m1]][v]=xr.DataArray(data=f_avg_all.T,coords={'hour':hour_avg,'height':ALL_sel.height.values}).interpolate_na(dim='height',limit=2).interpolate_na(dim='hour',limit=1)

#daily average (wind sector)
for wd1,wd2 in zip(bin_wd[:-1],bin_wd[1:]):
    if wd1>wd2:
        ALL_sel=ALL.where(((ALL['WD']>=wd1) & (ALL['WD']<360)) | ((ALL['WD']>=0) & (ALL['WD']<wd2)))
    else:
        ALL_sel=ALL.where(ALL['WD']>=wd1).where(ALL['WD']<wd2)
    ALL_avg[wd_name[wd1]]=xr.Dataset()
    for v in variables:
        f_avg_all=[]
        for h in height:
            f=ALL_sel[v].sel(height=h).values
            f[f==0]=np.nan
            real=~np.isnan(f)
            
            if np.sum(real)>0:
                f_avg= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_stat(x,np.nanmean,perc_lim=perc_lim),                             bins=bin_hour)[0]
                f_low= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100),    bins=bin_hour)[0]
                f_top= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=bin_hour)[0]
    
                f_avg[np.abs(f_top-f_low)>max_err[v]]=np.nan
                f_avg[np.isnan(np.abs(f_top-f_low))]=np.nan
                
                f_avg_all=utl.vstack(f_avg_all,f_avg)

            else:
                f_avg_all=utl.vstack(f_avg_all,hour_avg*np.nan)
            print(f'{v}: directional avg at {wd1} at {h} m done')
           
        ALL_avg[wd_name[wd1]][v]=xr.DataArray(data=f_avg_all.T,coords={'hour':hour_avg,'height':ALL_sel.height.values}).interpolate_na(dim='height',limit=2).interpolate_na(dim='hour',limit=1)

#%% Plots
plt.close('all')

# matplotlib.rcParams['font.size'] = 20
# all
fig=plt.figure(figsize=(16,4))
ctr=0
for v in variables:
    ax=plt.subplot(1,len(variables),ctr+1)
    cf=plt.contourf(hour_avg,ALL_avg['tot'].height,ALL_avg['tot'][v].T,ticks[v],cmap='coolwarm',extend='both')
    plt.contour(hour_avg,ALL_avg['tot'].height,ALL_avg['tot'][v].T,ticks[v],colors='k',linewidths=0.5,linestyles='solid',alpha=0.5)
    plt.ylabel(r'$z$ [m a.g.l.]')
    plt.xlabel('Hour (UTC)')
    ax.set_xticks([0,6,12,18,24])
    ax.set_xticklabels(['0000','0600','1200','1800','2400'])
    cbar=fig.colorbar(cf,label=labels[v])
    cbar.set_ticks(ticks[v][::2])
    cbar.set_ticklabels(ticklabels[v][::2])
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    ctr+=1
  
#special TKE figure
# matplotlib.rcParams['font.size'] = 16
fig=plt.figure(figsize=(16,8))
v='TKE log'
ax=plt.subplot(2,2,4)
cf=plt.contourf(hour_avg,ALL_avg['tot'].height,ALL_avg['tot'][v].T,ticks[v],cmap='coolwarm',extend='both')
plt.contour(hour_avg,ALL_avg['tot'].height,ALL_avg['tot'][v].T,ticks[v],colors='k',linewidths=0.5,linestyles='solid',alpha=0.5)
plt.ylabel(r'$z$ [m a.g.l.]')
ax.set_xticks([0,6,12,18,24])
ax.set_xticklabels(['0000','0600','1200','1800','2400'])
plt.xlabel('Hour (UTC)')
plt.xlim([0,24])
plt.ylim([0,2000])
# ax.set_yscale('symlog',linthresh=100)
cbar=plt.colorbar(cf,label=labels[v])
cbar.set_ticks(ticks[v])
cbar.set_ticklabels(ticklabels[v])
plt.tight_layout()
plt.grid()

#month/direction
# matplotlib.rcParams['font.size'] = 16
for v in variables:
    fig=plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(len(month_name), 3, width_ratios=[1,1,0.05])  # 0.1 row for the colorbars
    ctr=0
    for m in month_name:
        mn=month_name[m]
        ax=fig.add_subplot(gs[ctr,0])
        cf=plt.contourf(hour_avg,ALL_avg[mn].height,ALL_avg[mn][v].T,ticks[v],cmap='coolwarm',extend='both')
        plt.contour(hour_avg,ALL_avg[mn].height,ALL_avg[mn][v].T,ticks[v],colors='k',linewidths=0.5,linestyles='solid',alpha=0.5)
        
        plt.ylabel(r'$z$ [m a.g.l.]')
        plt.text(0.3,1600,mn,fontweight='bold',bbox={'alpha':0.25,'facecolor':'w'})
   
        ax.set_xticks([0,6,12,18,24])
        ax.set_xticklabels(['0000','0600','1200','1800','2400'])
        if ctr==len(month_name)-1:
            plt.xlabel('Hour (UTC)')
        else:
            ax.set_xticklabels([])
        plt.grid()
        ctr+=1
        
    ctr=0
    for w in wd_name:
        wdn=wd_name[w]
        ax=fig.add_subplot(gs[ctr,1])
        cf=plt.contourf(hour_avg,ALL_avg[wdn].height,ALL_avg[wdn][v].T,ticks[v],cmap='coolwarm',extend='both')
        plt.contour(hour_avg,ALL_avg[wdn].height,ALL_avg[wdn][v].T,ticks[v],colors='k',linewidths=0.5,linestyles='solid',alpha=0.5)
        plt.text(0.3,1600,wdn,fontweight='bold',bbox={'alpha':0.25,'facecolor':'w'})
        ax.set_xticks([0,6,12,18,24])
        ax.set_xticklabels(['0000','0600','1200','1800','2400'])
        if ctr==len(month_name)-1:
            plt.xlabel('Hour (UTC)')
        else:
            ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.grid()
        ctr+=1
        
    cbar_ax = ax=fig.add_subplot(gs[:,2])
    cbar=fig.colorbar(cf, cax=cbar_ax,label=labels[v])
    cbar.set_ticks(ticks[v])
    cbar.set_ticklabels(ticklabels[v])
    cbar.ax.xaxis.set_tick_params(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    