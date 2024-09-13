# -*- coding: utf-8 -*-
"""
Wake analysis
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
import re

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source_config=os.path.join(cd,'config.yaml')
source_log='data/20230101.000500-20240101.224500.awaken.glob.summary.csv'

#dataset
channel='awaken/rt1.lidar.z02.a0'
regex='\d{8}\.\d{1}\d*[13579]20\d{2}\.user5.nc'

#stats
max_TI=50#[%] maximum TI
TI_bin=[0,10]
WS_bin=[5,8]
WD_bin=[160,200]

#%% Initialization

#imports
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
sys.path.append(config['path_utils'])
sys.path.append(config['path_dap'])
sys.path.append(config['path_lidargo'])
import utils as utl
from doe_dap_dl import DAP
import LIDARGO_standardize as lb0
import LIDARGO_statistics as lc0

#load log data
LOG=pd.read_csv(os.path.join(cd,source_log)).replace(-9999, np.nan)
tnum_log=np.array([utl.datenum(t,'%Y-%m-%d %H:%M:%S') for t in LOG['UTC Time']])
time_log=np.array([utl.num_to_dt64(t) for t in tnum_log])
month_log=np.array([int(t[5:7]) for t in LOG['UTC Time'].values])
LOG=LOG.set_index('UTC Time')

_filter = {
    'Dataset': channel,
    'date_time': {
        'between': ['20230801000000','20230901000000']
    },
    'file_type': 'nc',
    'ext1':'user5', 
}
a2e = DAP('a2e.energy.gov',confirm_downloads=False)

#%% Main

LOG['Rotor-averaged TI [%]']=LOG['Rotor-averaged TI [%]'].where(LOG['Rotor-averaged TI [%]']>0).where(LOG['Rotor-averaged TI [%]']<max_TI)

#list files
files_dap=a2e.search(_filter)
if files_dap is None:
    a2e.setup_two_factor_auth(username=config['username'], password=config['password'])
    files_dap=a2e.search(_filter)

#select files
tnum_file=[utl.datenum(f['date_time'],'%Y%m%d%H%M%S') for f in files_dap]

WS_file=np.interp(tnum_file,tnum_log,LOG['Hub-height wind speed [m/s]'])
c=np.interp(tnum_file,tnum_log,utl.cosd(LOG['Hub-height wind direction [degrees]']))
s=np.interp(tnum_file,tnum_log,utl.sind(LOG['Hub-height wind direction [degrees]']))
WD_file=utl.cart2pol(c,s)[1]
TI_file=np.interp(tnum_file,tnum_log,LOG['Rotor-averaged TI [%]'])

sel_vol=np.array([len(re.findall(regex, f['Filename']))>0 for f in files_dap])

sel_WS=(WS_file>=WS_bin[0])*(WS_file<WS_bin[1])
sel_WD=(WD_file>=WD_bin[0])*(WD_file<WD_bin[1])
sel_TI=(TI_file>=TI_bin[0])*(TI_file<TI_bin[1])
files_dap_sel=np.array(files_dap)[sel_WS*sel_WD*sel_TI*sel_vol]

dirname=(str(WS_bin)+str(WD_bin)+str(TI_bin)).replace(' ','').replace(',','.').replace('[','.').replace(']','')[1:]
a2e.download_files(list(files_dap_sel),os.path.join(config['path_data'],channel,dirname,replace=False))

files=glob.glob(os.path.join(config['path_data'],channel,dirname,'*nc'))
for f in files_dap_sel:
    try:
        lproc_b0 = lb0.LIDARGO(f,os.path.join(cd,config['path_config_b0']), verbose=True)
        lproc_b0.process_scan(f,replace=False,save_file=True)
        
        lproc_c0 = lc0.LIDARGO(lproc_b0.save_filename,os.path.join(cd,config['path_config_c0']), verbose=True)
        if 'config' in dir(lproc_c0):
            lproc_c0.process_scan(lproc_b0.save_filename,replace=False,save_file=True)
            
    except Exception as e:
        print(e)
        



#%% Plots
plt.figure()
plt.plot(LOG['Hub-height wind speed [m/s]'],LOG['Rotor-averaged TI [%]'],'.k',alpha=0.1)