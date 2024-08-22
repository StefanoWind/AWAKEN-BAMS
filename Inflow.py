# -*- coding: utf-8 -*-
"""
Produce conditional average of wind profiles
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
from scipy.fft import fft, fftshift, fftfreq
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source_config=os.path.join(cd,'config.yaml')
sources_log={'A1':  'data/20230101.000500-20240101.224500.awaken.sa1.summary.csv',
             'A2':  'data/20230207.203500-20230911.121500.awaken.sa2.summary.csv',
             'H':   'data/20230101.000500-20240101.224500.awaken.sh.summary.csv',
             'GLOB':'data/20230101.000500-20240101.224500.awaken.glob.summary.csv'}

#dataset
channels=['awaken/sa1.lidar.z03.c1',
          'awaken/sh.lidar.z02.c1',
          'awaken/sa2.sonic.z01.c0']

sdate='20230501'#start date
edate='20230508'#end date

#DAP
username='sletizia'
password='pass_DAP1506@'

#stats
WD_range=[-160,200]

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

#%% Main 

#download data
for channel in channels:
    _filter = {
        'Dataset': channel,
        'date_time': {
            'between': [sdate,edate]
        },
    }
    
    a2e.setup_basic_auth(username=config['username'], password=config['password'])
    
    utl.mkdir(os.path.join(cd,'data',channel))
    a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel),replace=False)
    
    
#log
IN={}
for s in sources_log:
    IN[s]=pd.read_csv(os.path.join(cd,sources_log[s]))
    IN[s].index=([utl.num_to_dt64(utl.datenum(t,'%Y-%m-%d %H:%M:%S')) for t in IN[s]['UTC Time']])
    IN[s]=IN[s].drop(columns=['UTC Time','LLJ flag'])
    
IN_sel=IN['A1'].where(IN['A1']['Hub-height wind direction [degrees]']>WD_range[0]).where(IN['A1']['Hub-height wind direction [degrees]']<WD_range[-1])
WS_sel=IN_sel['Hub-height wind speed [m/s]']

nans1=np.where(np.diff(np.isnan(WS_sel))==1)[0]
nans0=np.where(np.diff(np.isnan(WS_sel))==-1)[0]