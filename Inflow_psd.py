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
channel_lid='awaken/sa1.lidar.z03.c1'
channel_ast='awaken/sb.assist.z01.c0'

sdate='20230724'#start date
edate='20230801'#end date

#stats
WD_range=[90,270]#[deg] wind direction range
min_points_psd=144*2*2/3# minimum contiguous time series length
DT=600#[s] resampling period
max_TKE=10#[m^2/s^2] maximum TKE
DT_psd=3600*2 #[s] resolution of spectrum
max_T=102*3600#[s] maximum period
min_lwp=10#[g/m^2] for liquid water paths lower than this, remove clouds
max_gamma=5#maximum weight of the prior
max_rmsr=5#maximum rms of the retrieval

#user-defined
variables_lid=['WS','WD','TKE']
variables_ast=['temperature','rh']
height=np.arange(0,2001,25)

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
for channel in [channel_lid,channel_ast]:
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
IN=pd.read_csv(os.path.join(cd,source_log)).replace(-9999, np.nan)

#load lidar data
LID=xr.open_mfdataset(glob.glob(os.path.join(cd,'data',channel_lid,'*nc')))

#load assist data
AST=xr.open_mfdataset(glob.glob(os.path.join(cd,'data',channel_ast,'*nc')))

#zeroing
PSD=xr.Dataset()

#graphics
utl.mkdir(os.path.join(cd,'figures'))

#%% Main 

#lidar preprocessing

#select wind sector
tnum_in=np.array([utl.datenum(t,'%Y-%m-%d %H:%M:%S') for t in IN['UTC Time']])
tnum_lid=np.array([utl.dt64_to_num(t) for t in LID.time.values])  
WD_c=np.interp(tnum_lid,tnum_in,utl.cosd(IN['Hub-height wind direction [degrees]'].values))
WD_s=np.interp(tnum_lid,tnum_in,utl.sind(IN['Hub-height wind direction [degrees]'].values))
WD_int=utl.cart2pol(WD_c,WD_s)[1]%360
LID['WD_hh']=xr.DataArray(data=WD_int,coords={'time':LID.time})
if WD_range[1]>WD_range[0]:
    LID_sel=LID.where(LID['WD_hh']>WD_range[0]).where(LID['WD_hh']<WD_range[1])
else:
    LID_sel=LID.where((LID['WD_hh']<WD_range[1]) | (LID['WD_hh']>WD_range[0]))

#lidar qc
TKE_qc=LID_sel['TKE'].where(LID_sel['TKE']>0).where(LID_sel['TKE']<max_TKE)
original_nans=np.isnan(LID_sel['TKE'])
TKE_int=TKE_qc.chunk({"time": -1}).interpolate_na(dim='time',method='linear')
TKE_int=TKE_int.where(original_nans==False)
LID_sel['TKE']=TKE_int
LID_int=LID_sel.interp(height=height)

#assist preprocessing

#select wind sector
tnum_ast=np.array([utl.dt64_to_num(t) for t in AST.time.values])  
WD_c=np.interp(tnum_ast,tnum_in,utl.cosd(IN['Hub-height wind direction [degrees]'].values))
WD_s=np.interp(tnum_ast,tnum_in,utl.sind(IN['Hub-height wind direction [degrees]'].values))
WD_int=utl.cart2pol(WD_c,WD_s)[1]%360
AST['WD_hh']=xr.DataArray(data=WD_int,coords={'time':AST.time})
if WD_range[1]>WD_range[0]:
    AST_sel=AST.where(AST['WD_hh']>WD_range[0]).where(AST['WD_hh']<WD_range[1])
else:
    AST_sel=AST.where((AST['WD_hh']<WD_range[1]) | (AST['WD_hh']>WD_range[0]))

#fix units
AST_sel['height']=AST_sel.height*1000
AST_sel['cbh']=AST_sel.cbh*1000

#assist qc
AST_sel['cbh'][AST_sel['lwp']<min_lwp]=10000
AST_sel=AST_sel.where(AST_sel['height']<AST_sel['cbh']).where(AST_sel['rmsr']<max_rmsr).where(AST_sel['gamma']<max_gamma)
AST_int=AST_sel.interp(height=height).interp(time=LID_int.time)

for v in variables_lid+variables_ast:
    psd_T=[]
    for h in height:
        print(v+': z = '+str(h)+' m')
        
        if v in variables_lid:
            f_sel=LID_int[v].sel(height=h)
        elif v in variables_ast:
                f_sel=AST_int[v].sel(height=h)
                
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
    plt.xlabel(r'Period [hour]')
    plt.ylabel(r'$z$ [m AGL]')
    plt.colorbar(label='Normalized PSD [$hour^{-1}$]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(cd,f'figures/{WD_range[0]}-{WD_range[1]}.psd.{v}.png'))
    plt.close()
    
#%% Output
PSD.to_netcdf(os.path.join(cd,f'data/{WD_range[0]}-{WD_range[1]}.psd.nc'))