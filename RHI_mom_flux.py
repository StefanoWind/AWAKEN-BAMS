# -*- coding: utf-8 -*-
"""
RHI processing
"""
import os
cd=os.path.dirname(__file__)
import utils as utl
import sys
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
import scipy as sp
import yaml
import re
import glob
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['savefig.dpi'] = 200

#%% Inputs
if len(sys.argv)==1:
    source_config=os.path.join(cd,'configs/config.yaml')
    ws_lim=[10.0,30.0]#m/s
    wd_lim=20.0#[deg]
    ti_lim=[0.0,10.0]#[deg]
    llj_lim=[300.0,500.0]
else:
    source_config=sys.argv[1]
    ws_lim=[np.float64(sys.argv[2]),np.float64(sys.argv[3])]
    wd_lim=np.float64(sys.argv[4])
    ti_lim=[np.float64(sys.argv[5]),np.float64(sys.argv[6])]
    llj_lim=[np.float64(sys.argv[7]),np.float64(sys.argv[8])]

#fixed inputs
source_log=os.path.join(cd,'data/glob.lidar.eventlog.avg.c2.20230101.000500.csv')#inflow table source
wd_ref=180#[deg]
min_cos=0.3
scan_duration=600#[s]
ele_corr=2

#stats
perc_lim=[5,95]#[%] percentile limits
p_value=0.05#p-value for c.i.
dx=100#[m]
dz=50#[m]
max_err_u=0.1
min_N=100
min_u=0.3
max_u=3

#graphics
max_plot=1000000

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

save_name=os.path.join(cd,'data',f'rhi.{ws_lim[0]}.{ws_lim[1]}.{wd_lim}.{ti_lim[0]}.{ti_lim[1]}.{llj_lim[0]}.{llj_lim[1]}.nc')
os.makedirs(os.path.join(cd,'figures',os.path.basename(save_name)[:-3]),exist_ok=True)
             
#%% Main

if not os.path.isfile(save_name):
    for s in config['source_rhi']:
        files=np.array(sorted(glob.glob(config['source_rhi'][s])))
        if len(files)>0:
            time_files=dates_from_files(files)
            
            #inflow extraction
            ws_int1=inflow['Nose wind speed [m/s]'].interp(time=time_files).values
            ws_int2=inflow['Nose wind speed [m/s]'].interp(time=time_files+np.timedelta64(scan_duration,'s')).values
            
            cos1=np.cos(np.radians(inflow['Hub-height wind direction [degrees]'])).interp(time=time_files).values
            sin1=np.sin(np.radians(inflow['Hub-height wind direction [degrees]'])).interp(time=time_files).values
            wd_int1=np.degrees(np.arctan2(sin1,cos1))%360
            cos2=np.cos(np.radians(inflow['Hub-height wind direction [degrees]'])).interp(time=time_files+np.timedelta64(scan_duration,'s')).values
            sin2=np.sin(np.radians(inflow['Hub-height wind direction [degrees]'])).interp(time=time_files+np.timedelta64(scan_duration,'s')).values
            wd_int2=np.degrees(np.arctan2(sin2,cos2))%360
            
            ti_int1=inflow['Rotor-averaged TI [%]'].interp(time=time_files).values
            ti_int2=inflow['Rotor-averaged TI [%]'].interp(time=time_files+np.timedelta64(scan_duration,'s')).values
            
            llj_int1=inflow['Nose height [m]'].interp(time=time_files).values
            llj_int2=inflow['Nose height [m]'].interp(time=time_files+np.timedelta64(scan_duration,'s')).values
            
            #file selection
            sel_ws=(ws_int1>=ws_lim[0])*(ws_int1<ws_lim[1])*(ws_int2>=ws_lim[0])*(ws_int2<ws_lim[1])
            
            wd_diff1=(wd_int1 - wd_ref + 180) % 360 - 180
            wd_diff2=(wd_int2 - wd_ref + 180) % 360 - 180
            sel_wd=(np.abs(wd_diff1)<wd_lim)*(np.abs(wd_diff2)<wd_lim)
           
            sel_ti=(ti_int1>=ti_lim[0])*(ti_int1<ti_lim[1])*(ti_int2>=ti_lim[0])*(ti_int2<ti_lim[1])
            
            sel_llj=(llj_int1>=llj_lim[0])*(llj_int1<llj_lim[1])*(llj_int2>=llj_lim[0])*(llj_int2<llj_lim[1])
            
            files_sel=files[sel_ws*sel_wd*sel_ti*sel_llj]
            
            print(f'{len(files_sel)} scans selected in {s}', flush=True)
            
            for f in files_sel:
                Data=xr.open_mfdataset(f)
                Data=Data.where(Data.qc_wind_speed==0).where(np.abs(np.cos(np.radians(Data.elevation)))>min_cos)
                real=~np.isnan(Data.x+Data.z+Data.wind_speed).values
                x=np.append(x,Data.x.values[real]+config['turbine_x'][s])
                z=np.append(z,Data.z.values[real])
                
                u_eq=-Data.wind_speed/np.cos(np.radians(Data.elevation+ele_corr))/((ws_int1[files==f]+ws_int2[files==f])/2)
                
                u=np.append(u,u_eq.values[real])
                
                #plot
                plt.figure(figsize=(18,4))
                plt.title(os.path.basename(f))
                ax=plt.gca()
                ax.set_aspect('equal')
                plt.xlim([-2000,8000])
                plt.ylim([0,1250])
                plt.xlabel(r'$x$ [m]')
                plt.ylabel(r'$y$ [m]')
                plt.grid()
                plt.colorbar(label='$u/U_\infty$ [m s$^{-1}$]')
                
                plt.savefig(os.path.join(cd,'figures',os.path.basename(save_name)[:-3],os.path.basename(f).replace('nc','png')))
                plt.close()
                
    #output
    Output=xr.Dataset()
    Output['x']=xr.DataArray(x,coords={'index':np.arange(len(x))})
    Output['z']=xr.DataArray(z,coords={'index':np.arange(len(x))})
    Output['u']=xr.DataArray(u,coords={'index':np.arange(len(x))})
    Output.to_netcdf(save_name)
    Output.close()

#load data
Data=xr.open_dataset(save_name)
Data=Data.where((Data.u>=min_u)*(Data.u<=max_u))
#stats
bin_x=np.arange(-2000,8000,dx)
bin_z=np.arange(0,1250,dz)
u_avg=stats.binned_statistic_2d(Data.x.values,Data.z.values,Data.u.values,
                                statistic=lambda x: utl.filt_stat(x,   np.nanmean,perc_lim=perc_lim),bins=[bin_x,bin_z])[0]
u_low=stats.binned_statistic_2d(Data.x.values,Data.z.values,Data.u.values,
                                statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100,min_N=min_N),bins=[bin_x,bin_z])[0]
u_top=stats.binned_statistic_2d(Data.x.values,Data.z.values,Data.u.values,
                                statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100,min_N=min_N),bins=[bin_x,bin_z])[0]

u_avg[u_top-u_low>max_err_u]=np.nan

x_grid=(bin_x[:-1]+bin_x[1:])/2
z_grid=(bin_z[:-1]+bin_z[1:])/2

#inpainting
interp_limit = 5
valid_mask1 = ~np.isnan(u_avg)
distance1 = sp.ndimage.distance_transform_edt(~valid_mask1)
interp_mask1 = (np.isnan(u_avg)) & (distance1 <= interp_limit)
yy1, xx1 = np.indices(u_avg.shape)
points1 = np.column_stack((yy1[valid_mask1], xx1[valid_mask1]))
values1 = u_avg[valid_mask1]
interp_points1 = np.column_stack((yy1[interp_mask1], xx1[interp_mask1]))
interpolated_values1 = sp.interpolate.griddata(points1, values1, interp_points1, method='linear')
u_avg_inp = u_avg.copy()
u_avg_inp[interp_mask1] = interpolated_values1

#%% Plots
plt.close('all')
skip=int(len(Data.x)/max_plot)
plt.figure(figsize=(18,4))
plt.scatter(Data.x.values[::skip],Data.z.values[::skip],s=1,c=Data.u.values[::skip],cmap='coolwarm',vmin=0.25,vmax=1)
ax=plt.gca()
ax.set_aspect('equal')
plt.xlim([-2000,8000])
plt.ylim([0,1250])
plt.grid()

plt.figure(figsize=(18,4))
cf=plt.contourf(x_grid,z_grid,u_avg.T,np.arange(0.25,1.01,0.1),cmap='coolwarm',extend='both')
plt.contour(x_grid,z_grid,u_avg.T,np.arange(0.25,1.01,0.1),extend='both',linewidths=1,alpha=0.25,colors='k')
ax=plt.gca()
ax.set_aspect('equal')
plt.xlim([-2000,8000])
plt.ylim([0,1250])
plt.grid()
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'$y$ [m]')
plt.grid()
plt.colorbar(cf,label='$u/U_\infty$ [m s$^{-1}$]')

plt.figure(figsize=(18,4))
cf=plt.contourf(x_grid,z_grid,u_avg_inp.T,np.arange(0.25,1.01,0.05),cmap='coolwarm',extend='both')
plt.contour(x_grid,z_grid,u_avg_inp.T,np.arange(0.25,1.01,0.05),extend='both',linewidths=1,alpha=0.25,colors='k')
ax=plt.gca()
ax.set_aspect('equal')
plt.xlim([-2000,8000])
plt.ylim([0,1250])
plt.grid()
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'$y$ [m]')
plt.grid()
plt.colorbar(cf,label='$u/U_\infty$ [m s$^{-1}$]')