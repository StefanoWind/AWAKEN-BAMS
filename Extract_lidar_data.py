# -*- coding: utf-8 -*-
"""
Extract lidar inflow data
"""
import os
cd=os.path.dirname(__file__)
from doe_dap_dl import DAP
import glob
import yaml
import xarray as xr

#%% Inputs
source_config=os.path.join(cd,'configs/config.yaml')

#dataset
channels=['awaken/sa1.lidar.z03.c1',
          'awaken/sa2.lidar.z01.c1',
          'awaken/sh.lidar.z02.c1']
             
sdate='20230724'#start date
edate='20230801'#end date
max_height=2000#[m] maximum height

#%% Initialization

#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
a2e = DAP('a2e.energy.gov',confirm_downloads=False)

#download sonic data
a2e.setup_basic_auth(username=config['username'], password=config['password'])
for channel in channels:
    _filter = {
        'Dataset': channel,
        'date_time': {
            'between': [sdate,edate]
        },
        'file_type': 'nc'
    }
    
    os.makedirs(os.path.join(config['path_data'],channel),exist_ok=True)
    a2e.download_with_order(_filter, path=os.path.join(config['path_data'],channel),replace=False)

#%% Main

#load lidar data
LID={}
for channel in channels:
    LID[channel]=xr.open_mfdataset(glob.glob(os.path.join(config['path_data'],channel,'*nc'))).sel(height=slice(0,max_height+21))

#%% Output
for channel in channels:
    LID[channel].to_netcdf(os.path.join(cd,'data/'+channel.split('/')[1]+'.'+sdate+'.'+edate+'.nc'))
