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
import glob
import yaml
import pandas as pd
import xarray as xr
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib
from scipy import signal
sys.path.append('C:/ProgramData/Anaconda3/Lib/site-packages/windrose')
from windrose_v2 import WindroseAxes
from scipy import stats

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source_config=os.path.join(cd,'config.yaml')

#dataset
sources_log=['data/20230101.000500-20240101.224500.awaken.sa1.summary.csv',
             'data/20230101.000500-20240101.224500.awaken.sh.summary.csv']
channels_snc=['awaken/sa2.sonic.z01.c0','awaken/sa5.sonic.z01.c0']

wd_range_log={'data/20230101.000500-20240101.224500.awaken.sa1.summary.csv':[90,270],
              'data/20230101.000500-20240101.224500.awaken.sh.summary.csv':[270,90]}
wd_range_snc={'awaken/sa2.sonic.z01.c0':[0,180],
              'awaken/sa5.sonic.z01.c0':[180,360]}

sdate='20230724'#start date
edate='20230727'#end date

#stats
max_TKE=10#[m^2/s^2] maximum TKE
stab_class={'VS':[0,50],
            'S':[20,200],
            'NS':[200,500],
            'N1':[500,np.inf],
            'N2':[-np.inf,-500],
            'NU':[-500,-200],
            'U':[-200,-100],
            'VU':[-100,0]}

#%% Initialization

#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
sys.path.append(config['path_dap'])
import utils as utl
from doe_dap_dl import DAP

a2e = DAP('a2e.energy.gov',confirm_downloads=False)

#download data
a2e.setup_basic_auth(username=config['username'], password=config['password'])
for channel in channels_snc:
    _filter = {
        'Dataset': channel,
        'date_time': {
            'between': [sdate,edate]
        },
        'file_type':['nc','csv']
    }
    
    utl.mkdir(os.path.join(cd,'data',channel))
    a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel),replace=False)
    
#load log
LOG={}
for c in sources_log:
    LOG[c]=pd.read_csv(os.path.join(cd,c)).replace(-9999, np.nan)
    time_log=np.array([utl.num_to_dt64(utl.datenum(t,'%Y-%m-%d %H:%M:%S')) for t in LOG[c]['UTC Time']])
    LOG[c]['time']=time_log
    LOG[c]=LOG[c].set_index('time')
    
#load sonic data
SNC={}
for c in channels_snc:
    files_snc=glob.glob(os.path.join(cd,'data',c,'*csv'))
    dfs=[]
    for f in files_snc:
        df = pd.read_csv(f).iloc[1:,:]
        dfs.append(df)

    SNC[c] = pd.concat(dfs, ignore_index=True)
    y=SNC[c]['year'].values
    m=SNC[c]['month'].values
    d=SNC[c]['day'].values
    H=SNC[c]['hour'].values
    M=SNC[c]['minute'].values
    time_snc=np.array([],dtype='datetime64')
    for i in range(len(y)):
        dtime=datetime(int(y[i]),int(m[i]),int(d[i]),int(H[i]),int(M[i]))
        time_snc=np.append(time_snc,utl.num_to_dt64((dtime-datetime(1970, 1, 1)).total_seconds()))

    SNC[c]['time']=time_snc
    SNC[c]=SNC[c].set_index('time').apply(pd.to_numeric)
    

#graphics
utl.mkdir(os.path.join(cd,'figures'))

#%% Main 

#freestream log
LOG_sel={}
for c in sources_log:
    if wd_range_log[c][1]>wd_range_log[c][0]:
        LOG_sel[c]=LOG[c].where(LOG[c]['Hub-height wind direction [degrees]']>wd_range_log[c][0]).where(LOG[c]['Hub-height wind direction [degrees]']<wd_range_log[c][1])
    else:
        LOG_sel[c]=LOG[c].where((LOG[c]['Hub-height wind direction [degrees]']<wd_range_log[c][1]) | (LOG[c]['Hub-height wind direction [degrees]']>wd_range_log[c][0]))

#combine logs
LOG_cmb = pd.concat([LOG_sel[c] for c in LOG_sel.keys()])
LOG_cmb = LOG_cmb.groupby(LOG_cmb.index).mean()

#qc log
LOG_cmb=LOG_cmb.where(LOG_cmb['Rotor-averaged TKE [m^2/s^2]']>0).where(LOG_cmb['Rotor-averaged TKE [m^2/s^2]']<max_TKE)

#sonic preprocessing
tnum_in=np.array([utl.dt64_to_num(t) for t in LOG_cmb.index])
SNC_sel={}
for c in channels_snc:
    tnum_snc=np.array([utl.dt64_to_num(t) for t in SNC[c].index.values])  
    WD_c=np.interp(tnum_snc,tnum_in,utl.cosd(LOG_cmb['Hub-height wind direction [degrees]'].values))
    WD_s=np.interp(tnum_snc,tnum_in,utl.sind(LOG_cmb['Hub-height wind direction [degrees]'].values))
    SNC[c]['WD_hh']=utl.cart2pol(WD_c,WD_s)[1]%360
    
    if wd_range_snc[c][1]>wd_range_snc[c][0]:
        SNC_sel[c]=SNC[c].where(SNC[c]['WD_hh']>wd_range_snc[c][0]).where(SNC[c]['WD_hh']<wd_range_snc[c][1])
    else:
        SNC_sel[c]=SNC[c].where((SNC[c]['WD_hh']<wd_range_snc[c][1]) | (SNC[c]['WD_hh']>wd_range_snc[c][0]))

#combine logs
SNC_cmb = pd.concat([SNC_sel[c] for c in SNC_sel.keys()])
SNC_cmb = SNC_cmb.groupby(SNC_cmb.index).mean()

#qc sonic
SNC_cmb=SNC_cmb.where(SNC_cmb['QC flag']==0)

#stab classes
SNC_cmb['Stability class']='null'
for s in stab_class.keys():
    SNC_cmb['Stability class'][(SNC_cmb['Obukhov\'s length']>stab_class[s][0])*(SNC_cmb['Obukhov\'s length']<stab_class[s][1])]=s
SNC_cmb['Stability class']=SNC_cmb['Stability class'].replace('N1','N').replace('N2','N')

N_tot=stats.binned_statistic(SNC_cmb['hour'].where(SNC_cmb['Stability class']!='null'),
                             SNC_cmb['Stability class'].where(SNC_cmb['Stability class']!='null'),
                             statistic='count',bins=np.arange(-0.5,24,1))[0]


#%% Plots
plt.close('all')

matplotlib.rcParams['font.size'] = 24
cmap=matplotlib.cm.get_cmap('coolwarm')

#wind speed rose
real=~np.isnan(LOG_cmb['Hub-height wind speed [m/s]']+LOG_cmb['Hub-height wind direction [degrees]'])
ax = WindroseAxes.from_ax()
ax.bar(LOG_cmb['Hub-height wind direction [degrees]'][real], LOG_cmb['Hub-height wind speed [m/s]'][real], normed=True,opening=0.8,cmap=cmap,edgecolor="white",bins=((0,4,6,8,10,12)))
ax.set_rgrids(np.arange(0,16,5), np.arange(0,16,5))
ax.set_legend(units=r'm s$^{-1}$')

#tke rose
real=~np.isnan(LOG_cmb['Rotor-averaged TKE [m^2/s^2]']+LOG_cmb['Hub-height wind direction [degrees]'])
ax = WindroseAxes.from_ax()
ax.bar(LOG_cmb['Hub-height wind direction [degrees]'][real], LOG_cmb['Rotor-averaged TKE [m^2/s^2]'][real], normed=True,opening=0.8,cmap=cmap,edgecolor="white",bins=((0,0.1,0.5,1,2)))
ax.set_rgrids(np.arange(0,16,5), np.arange(0,16,5))
ax.set_legend(units=r'm$^{2}$ s$^{-2}$')

#stability histogram
matplotlib.rcParams['font.size'] = 16
colors = [cmap(i) for i in np.linspace(0,1,len(stab_class)-1)]
N_cum=0
ctr=0
plt.figure(figsize=(14,5))
for s in stab_class:
    if s!='N2':
        if s=='N1':
            s='N'
        N=stats.binned_statistic(SNC_cmb['hour'].where(SNC_cmb['Stability class']==s),
                                 SNC_cmb['Stability class'].where(SNC_cmb['Stability class']==s),
                                 statistic='count',bins=np.arange(-0.5,24,1))[0]
        plt.bar(np.arange(24),N/N_tot*100,label=s,bottom=N_cum,color=colors[ctr])
        N_cum+=N/N_tot*100
        ctr+=1
plt.xticks(np.arange(0,24))        
plt.yticks(np.arange(0,101,25)) 
plt.xlabel('Hour (UTC)')
plt.ylabel('Occurrence [%]')   
plt.grid()
plt.legend()