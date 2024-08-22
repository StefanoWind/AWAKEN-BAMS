# -*- coding: utf-8 -*-
"""
Calculated psd of inflow variables
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
from scipy import signal
from scipy import stats

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source_config=os.path.join(cd,'config.yaml')
source_log='data/20230101.000500-20240101.224500.awaken.sa1.summary.csv'

#dataset
channel='awaken/sa1.lidar.z03.c1'

sdate='20230501'#start date
edate='20230508'#end date

#stats
WD_range=[100,260]#[deg] wind direction range
min_points_psd=144*2*2/3# minimum contigius time series length
DT=600#resampling frequency
max_TKE=5#[m^2/s^2]
DT_psd=3600*2

#user-defined
variables=['WS','WD','TKE']
skip=10

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

bin_f=1/np.arange(3600,7*24*3600,DT_psd)[::-1]

f_avg=1/utl.mid(1/bin_f)

PSD=xr.Dataset()



#download data
_filter = {
    'Dataset': channel,
    'date_time': {
        'between': [sdate,edate]
    },
    'file_type':'nc'
}

a2e.setup_basic_auth(username=config['username'], password=config['password'])

utl.mkdir(os.path.join(cd,'data',channel))
a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel),replace=False)
    
#load log
IN=pd.read_csv(os.path.join(cd,source_log)).replace(-9999, np.nan)

#%lidar data
LID=xr.open_mfdataset(glob.glob(os.path.join(cd,'data',channel,'*nc')))

#%% Main 

#select wind direction range
tnum_in=np.array([utl.datenum(t,'%Y-%m-%d %H:%M:%S') for t in IN['UTC Time']])
tnum_lid=np.array([utl.dt64_to_num(t) for t in LID.time.values])  
WD_int=np.interp(tnum_lid,tnum_in,IN['Hub-height wind direction [degrees]'].values)
LID['WD_hh']=xr.DataArray(data=WD_int,coords={'time':LID.time})
LID_sel=LID.where(LID['WD_hh']>WD_range[0]).where(LID['WD_hh']<WD_range[-1])
height=LID_sel.height.values[::skip]


for v in variables:
    psd_avg=[]
    for h in height:
        print(h)
        f_sel=LID_sel[v].sel(height=h)
        if np.sum(~np.isnan(f_sel))>min_points_psd:
        
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
                    f=f_sel[r1:r2]
                    tnum= [utl.dt64_to_num(t) for t in f.time]
                    tnum_res=np.arange(np.min(tnum),np.max(tnum)+1,DT)
                    f_res=np.interp(tnum_res,tnum,f.values)
                    f_psd, psd = signal.periodogram(f_res, fs=1/(DT),  scaling='density')
                    
                    f_psd_all=np.append(f_psd_all,f_psd)
                    psd_all=np.append(psd_all,psd)
            if len(psd_all)>0:   
                psd_avg_dim=stats.binned_statistic(f_psd_all,psd_all,statistic='mean',bins=bin_f)[0]
                psd_avg_norm=psd_avg_dim/(np.nansum(psd_avg_dim*DT_psd/3600))
                
                psd_avg=utl.vstack(psd_avg,psd_avg_norm)

            else:
                psd_avg=utl.vstack(psd_avg,f_avg*np.nan)
        else:
            psd_avg=utl.vstack(psd_avg,f_avg*np.nan)
  
    PSD[v]=xr.DataArray(data=psd_avg.T,coords={'frequency':f_avg,'height':height},attrs={'description':'Normalized power spectral density','units':'('+LID_sel[v].attrs['units']+')^2 h'})
    plt.figure()
    plt.pcolor(1/f_avg/3600,height,np.log10(PSD[v].T))
    plt.xlabel(r'Period [h]')
    plt.ylabel(r'$z$ [m AGL]')
    plt.colorbar(label='psd ['+LID_sel[v].attrs['units']+')^2 h]')
    plt.grid()
    
    plt.savefig(os.path.join(cd,f'data/{WD_range[0]}-{WD_range[1]}.psd.{v}.png'))
    plt.close()
    
#%% Output
PSD.to_netcdf(os.path.join(cd,f'data/{WD_range[0]}-{WD_range[1]}.psd.nc'))