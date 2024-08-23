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
min_points_psd=144*2*2/3# minimum contiguous time series length
DT=600#[s] resampling period
max_TKE=10#[m^2/s^2] maximum TKE
DT_psd=3600*2 #[s] resolution of spectrum
max_T=102*3600#[s] maximum period

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

#frequency bins
bin_f=1/np.arange(3600,max_T+1,DT_psd)[::-1]
f_avg=1/utl.mid(1/bin_f)
T_avg=utl.mid(np.arange(3600,max_T+1,DT_psd))

PSD=xr.Dataset()

#download data
_filter = {
    'Dataset': channel,
    'date_time': {
        'between': [sdate,edate]
    },
    'file_type':'nc'
}

utl.mkdir(os.path.join(cd,'data',channel))
a2e.setup_basic_auth(username=config['username'], password=config['password'])
a2e.download_with_order(_filter, path=os.path.join(cd,'data',channel),replace=False)
    
#load log
IN=pd.read_csv(os.path.join(cd,source_log)).replace(-9999, np.nan)

#load lidar data
LID=xr.open_mfdataset(glob.glob(os.path.join(cd,'data',channel,'*nc')))

#graphics
utl.mkdir(os.path.join(cd,'figures'))

#%% Main 

#select wind direction range
tnum_in=np.array([utl.datenum(t,'%Y-%m-%d %H:%M:%S') for t in IN['UTC Time']])
tnum_lid=np.array([utl.dt64_to_num(t) for t in LID.time.values])  
WD_int=np.interp(tnum_lid,tnum_in,IN['Hub-height wind direction [degrees]'].values)
LID['WD_hh']=xr.DataArray(data=WD_int,coords={'time':LID.time})
LID_sel=LID.where(LID['WD_hh']>WD_range[0]).where(LID['WD_hh']<WD_range[-1])
height=LID_sel.height.values[::skip]

#TKE qc
TKE_qc=LID_sel['TKE'].where(LID_sel['TKE']>0).where(LID_sel['TKE']<max_TKE)
original_nans=np.isnan(LID_sel['TKE'])
TKE_int=TKE_qc.chunk({"time": -1}).interpolate_na(dim='time',method='linear')
TKE_int=TKE_int.where(original_nans==False)
LID_sel['TKE']=TKE_int

for v in variables:
    psd_T=[]
    for h in height:
        print(h)
        f_sel=LID_sel[v].sel(height=h)
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
                    tnum= [utl.dt64_to_num(t) for t in f.time]
                    tnum_res=np.arange(np.min(tnum),np.max(tnum)+1,DT)
                    f_res=np.interp(tnum_res,tnum,f.values)
                    f_psd, psd = signal.periodogram(f_res, fs=1/DT,  scaling='density')
                    
                    #psd (frequency)
                    f_psd_all=np.append(f_psd_all,f_psd)
                    psd_all=np.append(psd_all,psd/np.var(f_res))
                    
            if len(psd_all)>0:   
                
                #average frequency spectrum
                psd_avg=stats.binned_statistic(f_psd_all,psd_all,statistic='mean',bins=bin_f)[0]
                
                #temporal spectrum
                psd_T=utl.vstack(psd_T,(psd_avg*f_avg**2)[::-1])

            else:
                psd_T=utl.vstack(psd_T,T_avg*np.nan)
        else:
            psd_T=utl.vstack(psd_T,T_avg*np.nan)
  
    PSD[v]=xr.DataArray(data=psd_T.T,coords={'period':T_avg,'height':height},attrs={'description':'Normalized power spectral density','units':'s^-1'})
    plt.figure()
    plt.pcolor(T_avg/3600,height,np.log10(PSD[v].T))
    plt.xlabel(r'Period [h]')
    plt.ylabel(r'$z$ [m AGL]')
    plt.colorbar(label='Normalized PSD [$h^{-1}$]')
    plt.grid()
    
    plt.savefig(os.path.join(cd,f'figures/{WD_range[0]}-{WD_range[1]}.psd.{v}.png'))
    plt.close()
    
#%% Output
PSD.to_netcdf(os.path.join(cd,f'data/{WD_range[0]}-{WD_range[1]}.psd.nc'))