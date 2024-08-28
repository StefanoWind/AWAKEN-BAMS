# -*- coding: utf-8 -*-
"""
Make daily met files
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

#dataset
channel='awaken/sa1.met.z01.b0'
sdate='20230724'#start date
edate='20230801'#end date

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

_filter = {
    'Dataset': channel,
    'date_time': {
        'between': [sdate,edate]
    },
    'file_type':['nc','csv']
}

utl.mkdir(os.path.join(config['path_data'],channel))
a2e.download_with_order(_filter, path=os.path.join(config['path_data'],channel),replace=False)

utl.mkdir(os.path.join(config['path_data'],channel.replace('b0','c0')))

#%% Main
dates=[utl.datestr(t,'%Y%m%d') for t in np.arange(utl.datenum(sdate,'%Y%m%d'),utl.datenum(edate,'%Y%m%d')+1,3600*24)]

for d in dates:
    Data=xr.open_mfdataset(glob.glob(os.path.join(config['path_data'],channel,'*'+d+'*nc')))
    bin_time=np.arange(Data.time.values[0],Data.time.values[-1]+np.timedelta64(10, 'm')/2,np.timedelta64(10, 'm'))
    Data_10m=Data.groupby_bins("time", bin_time).mean()
    Data_10m['time_bins']=np.arange(Data.time.values[0]+np.timedelta64(10, 'm')/2,Data.time.values[-1],np.timedelta64(10, 'm'))
    Data_10m=Data_10m.rename({"time_bins": "time"})
    Data_10m.to_netcdf(os.path.join(config['path_data'],channel.replace('b0','c0'),channel.replace('awaken/','').replace('b0','c0')+'.'+d+'.000000.nc'))
    print(d+' done')