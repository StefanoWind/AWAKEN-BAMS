# -*- coding: utf-8 -*-
"""
Calculated psd of inflow variables from met
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
from datetime import datetime
from scipy import signal
from scipy import stats

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source_config=os.path.join(cd,'config.yaml')
source_log='data/20230101.000500-20240101.224500.awaken.sa1.summary.csv'

#dataset
channel_met='awaken/sa1.met.z01.b0'
channel_snc='awaken/sa1.sonic.z01.c0'

sdate='20230724'#start date
edate='20230801'#end date

#stats
WD_range=[90,270]#[deg] wind direction range
min_points_psd=144# minimum contiguous time series length
DT=600#[s] resampling period
DT_psd=3600*2 #[s] resolution of spectrum
max_T=102*3600#[s] maximum period

#user-defined
variables_met=['average_wind_speed','wind_direction','temperature','relative_humidity']
variables_snc=['TKE']

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

#frequency bins
bin_f=1/np.arange(3600,max_T+1,DT_psd)[::-1]
f_avg=1/utl.mid(1/bin_f)
T_avg=utl.mid(np.arange(3600,max_T+1,DT_psd))

#download data
a2e.setup_basic_auth(username=config['username'], password=config['password'])
for channel in [channel_met,channel_snc]:
    _filter = {
        'Dataset': channel,
        'date_time': {
            'between': [sdate,edate]
        },
        'file_type':['nc','csv']
    }
    
    utl.mkdir(os.path.join(config['path_data'],channel))
    a2e.download_with_order(_filter, path=os.path.join(config['path_data'],channel),replace=False)
    
#load log
IN=pd.read_csv(os.path.join(cd,source_log)).replace(-9999, np.nan)

#load met data
MET=xr.open_mfdataset(glob.glob(os.path.join(config['path_data'],channel_met,'*nc')))

#load sonic data
files_snc=glob.glob(os.path.join(config['path_data'],channel_snc,'*csv'))
dfs=[]
for f in files_snc:
    df = pd.read_csv(f).iloc[1:,:]
    dfs.append(df)
SNC_df = pd.concat(dfs, ignore_index=True)

#zeroing
PSD=xr.Dataset()

#graphics
utl.mkdir(os.path.join(cd,'figures'))

#%% Main 

#met preprocessing
MET['time_bin'] = pd.cut(MET['time'], bins=np.arange(MET.time.values[0],MET.time.values[-1]+np.timedelta64(10, 'm')/2,np.timedelta64(10, 'm')))
MET_10m=MET.groupby('time_bin').mean().reset_index()


#select wind sector
tnum_in=np.array([utl.datenum(t,'%Y-%m-%d %H:%M:%S') for t in IN['UTC Time']])
tnum_met=np.array([utl.dt64_to_num(t) for t in MET.time.values])  
WD_c=np.interp(tnum_met,tnum_in,utl.cosd(IN['Hub-height wind direction [degrees]'].values))
WD_s=np.interp(tnum_met,tnum_in,utl.sind(IN['Hub-height wind direction [degrees]'].values))
WD_int=utl.cart2pol(WD_c,WD_s)[1]%360
MET['WD_hh']=xr.DataArray(data=WD_int,coords={'time':MET.time})
if WD_range[1]>WD_range[0]:
    MET_sel=MET.where(MET['WD_hh']>WD_range[0]).where(MET['WD_hh']<WD_range[1])
else:
    MET_sel=MET.where((MET['WD_hh']<WD_range[1]) | (MET['WD_hh']>WD_range[0]))
    
#qc met
for v in variables_met:
    MET_sel[v]=MET_sel[v].where(MET_sel['qc_'+v]==0)

#sonic preprocessoning
y=SNC_df['year'].values
m=SNC_df['month'].values
d=SNC_df['day'].values
H=SNC_df['hour'].values
M=SNC_df['minute'].values
time_snc=np.array([],dtype='datetime64')
for i in range(len(y)):
    dtime=datetime(int(y[i]),int(m[i]),int(d[i]),int(H[i]),int(M[i]))
    time_snc=np.append(time_snc,utl.num_to_dt64((dtime-datetime(1970, 1, 1)).total_seconds()))

SNC_df['time']=time_snc
SNC_df=SNC_df.set_index('time').apply(pd.to_numeric)
SNC=xr.Dataset.from_dataframe(SNC_df.apply(pd.to_numeric))
tnum_snc=np.array([utl.dt64_to_num(t) for t in SNC.time.values])  
WD_c=np.interp(tnum_snc,tnum_in,utl.cosd(IN['Hub-height wind direction [degrees]'].values))
WD_s=np.interp(tnum_snc,tnum_in,utl.sind(IN['Hub-height wind direction [degrees]'].values))
WD_int=utl.cart2pol(WD_c,WD_s)[1]%360
SNC['WD_hh']=xr.DataArray(data=WD_int,coords={'time':SNC.time})
if WD_range[1]>WD_range[0]:
    SNC_sel=SNC.where(SNC['WD_hh']>WD_range[0]).where(SNC['WD_hh']<WD_range[1])
else:
    SNC_sel=SNC.where((SNC['WD_hh']<WD_range[1]) | (SNC['WD_hh']>WD_range[0]))

#qc sonic
SNC_sel=SNC_sel.where(SNC_sel['QC flag']==0)

#psd
variables=variables_snc+variables_met
for v in variables:
    print(v)

    if v in variables_met:
        f_sel=MET_sel[v]
    elif v in variables_snc:
            f_sel=SNC_sel[v]
        
    if np.sum(~np.isnan(f_sel))>min_points_psd:
    
        #find real values
        real2=np.where(np.diff(np.isnan(f_sel)+0)==1)[0]
        real1=np.where(np.diff(np.isnan(f_sel)+0)==-1)[0]+1
    
        if ~np.isnan(f_sel[0]):
            real1=np.append(0,real1)
        if ~np.isnan(f_sel[-1]):
            real2=np.append(real2,len(f_sel))

        f_psd_all=[]
        psd_all=[]
        for r1,r2 in zip(real1,real2):
            if r2-r1>min_points_psd:
            
                #resampling
                f=f_sel[r1:r2]
                tnum= np.array([utl.dt64_to_num(t) for t in f.time])
                tnum_res=np.arange(np.min(tnum),np.max(tnum)+1,DT)
                f_res=np.interp(tnum_res,tnum,f.values)
                f_psd, psd = signal.periodogram(f_res, fs=1/DT,  scaling='density')
            
                #psd (frequency)
                f_psd_all=np.append(f_psd_all,f_psd)
                psd_all=np.append(psd_all,psd/np.var(f_res))
            
        if len(psd_all)>0:   
            raise BaseException()
            #average frequency spectrum
            psd_avg=stats.binned_statistic(f_psd_all,psd_all,statistic='mean',bins=bin_f)[0]
        
            #temporal spectrum
            psd_T=(psd_avg*f_avg**2)[::-1]

        else:
            psd_T=T_avg*np.nan
    else:
        psd_T=T_avg*np.nan
  
    PSD[v]=xr.DataArray(data=psd_T.T,coords={'period':T_avg},attrs={'description':'Normalized power spectral density','units':'s^-1'})
    plt.figure()
    plt.loglog(T_avg/3600,PSD[v].T,'.-k')
    plt.xlabel(r'Period [hour]')
    plt.ylabel('Normalized PSD [hour$^{-1}$]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(cd,f'figures/{WD_range[0]}-{WD_range[1]}.met.psd.{v}.png'))
    plt.close()

#%% Output
PSD.to_netcdf(os.path.join(cd,f'data/{WD_range[0]}-{WD_range[1]}.met.psd.nc'))