# -*- coding: utf-8 -*-
"""
Extract met and sonic variables
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

#select wind sector
tnum_in=np.array([utl.datenum(t,'%Y-%m-%d %H:%M:%S') for t in IN['UTC Time']])
tnum_met=np.array([utl.dt64_to_num(t) for t in MET.time.values])  
WD_c=np.interp(tnum_met,tnum_in,utl.cosd(IN['Hub-height wind direction [degrees]'].values))
WD_s=np.interp(tnum_met,tnum_in,utl.sind(IN['Hub-height wind direction [degrees]'].values))
WD_int=utl.cart2pol(WD_c,WD_s)[1]%360
MET['WD_hh']=xr.DataArray(data=WD_int,coords={'time':MET.time})

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


#%% Output
MET.to_netcdf(os.path.join(cd,'data/met.nc'))
SNC.to_netcdf(os.path.join(cd,'data/snc.nc'))