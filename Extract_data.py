# -*- coding: utf-8 -*-
"""
Extract lidar, ASSIST, met and sonic variables
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

#dataset
channel_lid='awaken/sa1.lidar.z03.c1'
channel_ast='awaken/sb.assist.z01.c0'
source_met='awaken/sa2.met.z01.c0'
channel_snc='awaken/sa2.sonic.z01.c0'

sdate='20230724'#start date
edate='20230801'#end date

#user defined
height=np.array([0,110.001,200,500,1000,2000])#[m] height for remote sesing data extraction

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

#download sonic data
a2e.setup_basic_auth(username=config['username'], password=config['password'])
for channel in [channel_lid,channel_ast,channel_snc]:
    _filter = {
        'Dataset': channel,
        'date_time': {
            'between': [sdate,edate]
        },
        'file_type':['nc','csv']
    }
    
    utl.mkdir(os.path.join(config['path_data'],channel))
    a2e.download_with_order(_filter, path=os.path.join(config['path_data'],channel),replace=False)

#load lidar data
LID=xr.open_mfdataset(glob.glob(os.path.join(config['path_data'],channel_lid,'*nc')))

#load assist data
AST=xr.open_mfdataset(glob.glob(os.path.join(config['path_data'],channel_ast,'*nc')))

#load met data
MET=xr.open_mfdataset(glob.glob(os.path.join(config['path_data'],source_met,'*nc')))

#load sonic data
files_snc=glob.glob(os.path.join(config['path_data'],channel_snc,'*csv'))
dfs=[]
for f in files_snc:
    df = pd.read_csv(f).iloc[1:,:]
    dfs.append(df)
SNC_df = pd.concat(dfs, ignore_index=True)

#zeroing
PSD=xr.Dataset()

#%% Main 

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

#interpolation
LID_int=LID.interp(height=height)
AST_int=AST.interp(height=height/1000)

#%% Output
LID_int.to_netcdf(os.path.join(cd,'data/'+channel_lid.split('/')[1]+'.'+sdate+'.'+edate+'.nc'))
AST_int.to_netcdf(os.path.join(cd,'data/'+channel_ast.split('/')[1]+'.'+sdate+'.'+edate+'.nc'))
MET.to_netcdf(os.path.join(cd,'data/'+source_met.split('/')[1]+'.'+sdate+'.'+edate+'.nc'))
SNC.to_netcdf(os.path.join(cd,'data/'+channel_snc.split('/')[1]+'.'+sdate+'.'+edate+'.nc'))