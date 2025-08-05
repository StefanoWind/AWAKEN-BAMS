# -*- coding: utf-8 -*-
"""
Windroses and stability plots
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
sys.path.append('C:/ProgramData/Anaconda3/Lib/site-packages/windrose')
from windrose_v2 import WindroseAxes
from scipy import stats

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source_config=os.path.join(cd,'config.yaml')

#dataset
source_log='data/20230101.000500-20240101.224500.awaken.glob.summary.csv'
sources_snc={'A2':'data/sa2.sonic.z01.c0.20230101.20240101.nc',
             'A5':'data/sa5.sonic.z01.c0.20230101.20240101.nc'}

wd_range_snc={'A2':[0,270],
              'A5':[270,360]}

#stats
max_TKE=10#[m^2/s^2] maximum TKE
stab_class={'VS':[0,50],
            'S':[50,200],
            'NS':[200,500],
            'N1':[500,np.inf],
            'N2':[-np.inf,-500],
            'NU':[-500,-200],
            'U':[-200,-100],
            'VU':[-100,0]}

bin_month=np.arange(0,12.1,3)
bin_wd=[315,45,135,225,315]#[deg]

#graphics
month_name={0:'DJF',3:'MAM',6:'JJA',9:'SON'}
wd_name={315:'N',45:'E',135:'S',225:'W'}

#%% Functions
def nanmean_dataset(ds1, ds2):
    avg_dict = {}
    ds1_synch,ds2_synch=xr.align(ds1,ds2,join='outer')
    for var in ds1.data_vars:
        avg_dict[var] = xr.DataArray(np.nanmean([ds1_synch[var], ds2_synch[var]], axis=0),
                                    dims=ds1_synch[var].dims, coords=ds1_synch[var].coords)
    return xr.Dataset(avg_dict)

#%% Initialization

#load log
LOG=pd.read_csv(os.path.join(cd,source_log)).replace(-9999, np.nan)
tnum_log=np.array([utl.datenum(t,'%Y-%m-%d %H:%M:%S') for t in LOG['UTC Time']])
time_log=np.array([utl.num_to_dt64(t) for t in tnum_log])
month_log=np.array([int(t[5:7]) for t in LOG['UTC Time'].values])
month_log[month_log==12]=0
wd=LOG['Hub-height wind direction [degrees]'].values
wd[wd==-9999]=np.nan
LOG=LOG.set_index('UTC Time')

#load sonic
SNC={}
SNC_unw={}
for s in sources_snc:
    SNC[s]=xr.open_dataset(os.path.join(cd,sources_snc[s]))
    SNC[s]=SNC[s].where(SNC[s]['QC flag']==0)
    
    #rejected sectors affected by container
    if wd_range_snc[s][1]>wd_range_snc[s][0]:
       SNC_unw[s]=SNC[s].where(SNC[s]['wind direction']>=wd_range_snc[s][0]).where(SNC[s]['wind direction']<wd_range_snc[s][1])
    else:
       SNC_unw[s]=SNC[s].where((SNC[s]['wind direction']<wd_range_snc[s][1]) | (SNC[s]['wind direction']>=wd_range_snc[s][0]))
    
#graphics
utl.mkdir(os.path.join(cd,'figures'))

#%% Main 

#qc log
LOG=LOG.where(LOG['Rotor-averaged TKE [m^2/s^2]']>0).where(LOG['Rotor-averaged TKE [m^2/s^2]']<max_TKE)

#combine logs
SNC_cmb = nanmean_dataset(SNC_unw[list(sources_snc.keys())[0]], SNC_unw[list(sources_snc.keys())[1]]).sortby('time')

#stab classes
SNC_cmb['Stability class']=xr.DataArray(data=['null']*len(SNC_cmb.time),coords={'time':SNC_cmb.time})

for s in stab_class.keys():
    sel=(SNC_cmb['Obukhov\'s length']>=stab_class[s][0])*(SNC_cmb['Obukhov\'s length']<stab_class[s][1])
    if s=='N1' or s=='N2':
        s='N'
    SNC_cmb['Stability class']=SNC_cmb['Stability class'].where(~sel,other=s)

SNC_cmb['month'][SNC_cmb['month']==12]=0#december to 0

#hub-height wind direction
tnum_snc=np.float64(SNC_cmb['time'].values-np.datetime64('1970-01-01T00:00:00'))/10**9
c=np.interp(tnum_snc,tnum_log,utl.cosd(wd))
s=np.interp(tnum_snc,tnum_log,utl.sind(wd))
SNC_cmb['WD']=xr.DataArray(data=utl.cart2pol(c,s)[1],coords={'time':SNC_cmb.time})

#%% Plots
plt.close('all')

matplotlib.rcParams['font.size'] = 16
cmap=matplotlib.cm.get_cmap('coolwarm')

#wind speed rose by month
fig=plt.figure(figsize=(10,10))
ctr=1
for m1,m2 in zip(bin_month[:-1],bin_month[1:]):
    ax0=fig.add_subplot(2,2,ctr)
    pos=ax0.get_position()
    ax0.spines['top'].set_color('white')
    ax0.spines['right'].set_color('white')
    ax0.spines['bottom'].set_color('white')
    ax0.spines['left'].set_color('white')
    ax0.set_xticks([])
    ax0.set_yticks([])
    plt.text(0,1,month_name[m1],fontweight='bold',bbox={'alpha':0.1,'facecolor':'w'})
    wd_sel=LOG['Hub-height wind direction [degrees]'][(month_log>=m1)*(month_log<m2)]
    ws_sel=LOG['Hub-height wind speed [m/s]'][(month_log>=m1)*(month_log<m2)]
    real=~np.isnan(wd_sel+ws_sel)
    ax = WindroseAxes.from_ax(fig=fig,rect=[pos.x0,pos.y0,pos.width,pos.height])
    ax.bar(wd_sel[real], ws_sel[real], normed=True,opening=0.8,cmap=cmap,edgecolor="white",bins=((0,4,6,8,10,12)))
    ax.set_rgrids(np.arange(0,16,5), [str(t)+'%' for t in np.arange(0,16,5)])
    if ctr==1:
        ax.set_legend(units=r'm s$^{-1}$')
    else:
        ax.set_legend(units=r'm s$^{-1}$').set_visible(False)
    
    ctr+=1

#tke rose by month
fig=plt.figure(figsize=(10,10))
ctr=1
for m1,m2 in zip(bin_month[:-1],bin_month[1:]):
    ax0=fig.add_subplot(2,2,ctr)
    pos=ax0.get_position()
    ax0.spines['top'].set_color('white')
    ax0.spines['right'].set_color('white')
    ax0.spines['bottom'].set_color('white')
    ax0.spines['left'].set_color('white')
    ax0.set_xticks([])
    ax0.set_yticks([])
    plt.text(0,1,month_name[m1],fontweight='bold',bbox={'alpha':0.1,'facecolor':'w'})
    wd_sel=LOG['Hub-height wind direction [degrees]'][(month_log>=m1)*(month_log<m2)]
    tke_sel=LOG['Rotor-averaged TKE [m^2/s^2]'][(month_log>=m1)*(month_log<m2)]
    real=~np.isnan(wd_sel+tke_sel)
    ax = WindroseAxes.from_ax(fig=fig,rect=[pos.x0,pos.y0,pos.width,pos.height])
    ax.bar(wd_sel[real], tke_sel[real], normed=True,opening=0.8,cmap=cmap,edgecolor="white",bins=((0,0.1,0.5,1,2)))
    ax.set_rgrids(np.arange(0,16,5), [str(t)+'%' for t in np.arange(0,16,5)])
    if ctr==1:
        ax.set_legend(units=r'm$^{2}$ s$^{-2}$')
    else:
        ax.set_legend(units=r'm$^{2}$ s$^{-2}$').set_visible(False)
    
    ctr+=1
    
#stability by month
matplotlib.rcParams['font.size'] = 16
colors = [cmap(i) for i in np.linspace(0,1,len(stab_class)-1)]

plt.figure(figsize=(16,10))
ctr1=1
for m1,m2 in zip(bin_month[:-1],bin_month[1:]):
    plt.subplot(len(month_name),1,ctr1)
    SNC_sel=SNC_cmb.where(SNC_cmb['month']>=m1).where(SNC_cmb['month']<m2)
    N_tot=stats.binned_statistic(SNC_sel['hour'].where(SNC_sel['Stability class']!='null'),
                                 SNC_sel['Stability class'].where(SNC_sel['Stability class']!='null'),
                                 statistic='count',bins=np.arange(-0.5,24,1))[0]
    N_cum=0
    ctr2=0   
    for s in stab_class:
        if s!='N2':
            if s=='N1':
                s='N'
            N=stats.binned_statistic(SNC_sel['hour'].where(SNC_sel['Stability class']==s),
                                      SNC_sel['Stability class'].where(SNC_sel['Stability class']==s),
                                      statistic='count',bins=np.arange(-0.5,24,1))[0]
            plt.bar(np.arange(24),N/N_tot*100,label=s,bottom=N_cum,color=colors[ctr2])
            N_cum+=N/N_tot*100
            ctr2+=1
    plt.xticks(np.arange(0,24))        
    plt.yticks(np.arange(0,101,25)) 
    
    plt.ylabel('Occurrence [%]')   
    plt.grid()
    plt.text(-1,80,month_name[m1],fontweight='bold',bbox={'alpha':0.1,'facecolor':'w'})
    ctr1+=1
plt.legend(draggable='True')
plt.xlabel('Hour (UTC)')

#stability by sector
plt.figure(figsize=(16,10))
ctr1=1
for wd1,wd2 in zip(bin_wd[:-1],bin_wd[1:]):
    if wd1>wd2:
        SNC_sel=SNC_cmb.where((SNC_cmb['WD']>=wd1) | (SNC_cmb['WD']<wd2))
    else:
        SNC_sel=SNC_cmb.where(SNC_cmb['WD']>=wd1).where(SNC_cmb['WD']<wd2)
    plt.subplot(len(month_name),1,ctr1)
    N_tot=stats.binned_statistic(SNC_sel['hour'].where(SNC_sel['Stability class']!='null'),
                                 SNC_sel['Stability class'].where(SNC_sel['Stability class']!='null'),
                                 statistic='count',bins=np.arange(-0.5,24,1))[0]
    N_cum=0
    ctr2=0   
    for s in stab_class:
        if s!='N2':
            if s=='N1':
                s='N'
            N=stats.binned_statistic(SNC_sel['hour'].where(SNC_sel['Stability class']==s),
                                     SNC_sel['Stability class'].where(SNC_sel['Stability class']==s),
                                     statistic='count',bins=np.arange(-0.5,24,1))[0]
            plt.bar(np.arange(24),N/N_tot*100,label=s,bottom=N_cum,color=colors[ctr2])
            N_cum+=N/N_tot*100
            ctr2+=1
    plt.xticks(np.arange(0,24))        
    plt.yticks(np.arange(0,101,25)) 
    
    plt.ylabel('Occurrence [%]')   
    plt.grid()
    plt.text(-1,80,wd_name[wd1],fontweight='bold',bbox={'alpha':0.1,'facecolor':'w'})
    
    ctr1+=1
plt.legend(draggable='True')
plt.xlabel('Hour (UTC)')

#overall stabily histogram
plt.figure(figsize=(16,4))
N_tot=stats.binned_statistic(SNC_cmb['hour'].where(SNC_cmb['Stability class']!='null'),
                         SNC_cmb['Stability class'].where(SNC_cmb['Stability class']!='null'),
                      statistic='count',bins=np.arange(-0.5,24,1))[0]
N_cum=0
ctr2=0   
for s in stab_class:
    if s!='N2':
        if s=='N1':
            s='N'
        N=stats.binned_statistic(SNC_cmb['hour'].where(SNC_cmb['Stability class']==s),
                                 SNC_cmb['Stability class'].where(SNC_cmb['Stability class']==s),
                                 statistic='count',bins=np.arange(-0.5,24,1))[0]
        plt.bar(np.arange(24),N/N_tot*100,label=s,bottom=N_cum,color=colors[ctr2])
        N_cum+=N/N_tot*100
        ctr2+=1
plt.xticks(np.arange(0,24))        
plt.yticks(np.arange(0,101,25)) 

plt.ylabel('Occurrence [%]')   
plt.grid()
plt.legend(draggable='True')
plt.xlabel('Hour (UTC)')

#check sonic wake
plt.figure(figsize=(18,8))
ctr=1
for s in sources_snc:
    plt.subplot(len(sources_snc)+1,1,ctr)
    plt.semilogy(SNC[s]['wind direction'],SNC[s]['TKE'],'.r',alpha=0.1,label='All data')
    plt.semilogy(SNC_unw[s]['wind direction'],SNC_unw[s]['TKE'],'.g',alpha=0.2,label='Unwaked data')
    plt.ylabel(r'TKE at '+s+' [m$^2$ s$^{-2}$]')
    plt.grid()
    plt.xlim([0,360])
    plt.ylim([0.01,20])
    ctr+=1
plt.subplot(len(sources_snc)+1,1,ctr)
plt.semilogy(SNC_cmb['wind direction'],SNC_cmb['TKE'],'.k',alpha=0.1)
plt.ylabel(r'Unwaked TKE'+s+' [m$^2$ s$^{-2}$]')
plt.ylabel(r'TKE at '+s+' [m$^2$ s$^{-2}$]')
plt.xlabel(r'$\theta_w$ [$^\circ$]')
plt.xlim([0,360])
plt.ylim([0.01,20])
plt.grid()
plt.tight_layout()
