# -*- coding: utf-8 -*-
"""
RHI processing
"""
import os
cd=os.path.dirname(__file__)
import sys
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
import yaml
import re
import glob
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['savefig.dpi'] = 500
#%% Inputs
if len(sys.argv)==1:
    source_config=os.path.join(cd,'configs/config.yaml')
    ws_lim=[4,11]#m/s
    wd_lim=180#[deg]
    ti_lim=[0,15]#[deg]
else:
    source_config=sys.argv[1]
    ws_lim=[np.float64(sys.argv[2]),np.float64(sys.argv[3])]
    wd_lim=np.float64(sys.argv[4])
    ti_lim=[np.float64(sys.argv[5]),np.float64(sys.argv[6])]

#fixed inputs
source_log=os.path.join(cd,'data/glob.lidar.eventlog.avg.c2.20230101.000500.csv')#inflow table source
wd_ref=180#[deg]
min_cos=0.3
dx=200#[m]
dz=50#[m]
max_plot=10000

#%% Functions
def dates_from_files(files):
    '''
    Extract data from data filenames
    '''
    dates=np.array([],dtype='datetime64')
    for f in files:
        match = re.search( r"\b\d{8}\.\d{6}\b", os.path.basename(f))
        datestr=match.group()
        dates=np.append(dates,np.datetime64(f'{datestr[:4]}-{datestr[4:6]}-{datestr[6:8]}T{datestr[9:11]}:{datestr[11:13]}:{datestr[13:15]}'))
    
    return dates

#%% Initialization
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

#inflow
inflow_df=pd.read_csv(source_log).set_index('UTC Time')
inflow_df.index= pd.to_datetime(inflow_df.index)
inflow=xr.Dataset.from_dataframe(inflow_df).rename({'UTC Time':'time'})

#zeroing
x=[]
z=[]
u=[]

save_name=os.path.join(cd,f'rhi.{ws_lim[0]}.{ws_lim[1]}.{wd_lim}.{ti_lim[0]}.{ti_lim[1]}.nc')

#%% Main

if not os.path.isfile(save_name):
    for s in config['source_rhi']:
        files=np.array(sorted(glob.glob(config['source_rhi'][s])))
        if len(files)>0:
            time_files=dates_from_files(files)
            time_files+=np.median(np.diff(time_files))/2
            
            #inflow extraction
            ws_int=inflow['Hub-height wind speed [m/s]'].interp(time=time_files)
            cos=np.cos(np.radians(inflow['Hub-height wind direction [degrees]'])).interp(time=time_files)
            sin=np.sin(np.radians(inflow['Hub-height wind direction [degrees]'])).interp(time=time_files)
            wd_int=np.degrees(np.arctan2(sin,cos))%360
            ti_int=inflow['Rotor-averaged TI [%]'].interp(time=time_files)
            
            #file selection
            sel_ws=(ws_int>=ws_lim[0])*(ws_int<ws_lim[1])
            
            wd_diff=(wd_int - wd_ref + 180) % 360 - 180
            sel_wd=np.abs(wd_diff)<wd_lim
           
            sel_ti=(ti_int>=ti_lim[0])*(ti_int<ti_lim[1])
            
            files_sel=files[sel_ws*sel_wd*sel_ti]
            
            for f in files_sel:
                Data=xr.open_mfdataset(f)
                Data=Data.where(Data.qc_wind_speed==0).where(np.abs(np.cos(np.radians(Data.elevation)))>min_cos)
                real=~np.isnan(Data.x+Data.z+Data.wind_speed).values
                x=np.append(x,Data.x.values[real]+config['turbine_x'][s])
                z=np.append(z,Data.z.values[real])
                
                u_eq=-Data.wind_speed/np.cos(np.radians(Data.elevation))/ws_int.values[files==f]
                
                u=np.append(u,u_eq.values[real])
                
    #output
    Output=xr.Dataset()
    Output['x']=xr.DataArray(x,coords={'index':np.arange(len(x))})
    Output['z']=xr.DataArray(z,coords={'index':np.arange(len(x))})
    Output['u']=xr.DataArray(u,coords={'index':np.arange(len(x))})
    Output.to_netcdf(save_name)
    Output.close()

Data=xr.open_dataset(save_name)

#stats
bin_x=np.arange(-2000,8000,dx)
bin_z=np.arange(0,1000,dz)
u_avg=stats.binned_statistic_2d(Data.x.values,Data.z.values,Data.u.values,statistic='median',bins=[bin_x,bin_z])[0]

x_grid=(bin_x[:-1]+bin_x[1:])/2
z_grid=(bin_z[:-1]+bin_z[1:])/2

#%% Plots
skip=int(len(Data.x)/max_plot)
plt.figure(figsize=(18,4))
plt.scatter(Data.x.values[::skip],Data.z.values[::skip],s=1,c=Data.u.values[::skip],cmap='coolwarm',vmin=0.5,vmax=2)
ax=plt.gca()
ax.set_aspect('equal')
plt.xlim([-2000,8000])
plt.grid()

plt.figure(figsize=(18,4))
plt.pcolor(x_grid,z_grid,u_avg.T,cmap='coolwarm',vmin=0.5,vmax=2)
ax=plt.gca()
ax.set_aspect('equal')
plt.grid()