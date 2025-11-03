# -*- coding: utf-8 -*-
"""
Make video of RHI from NML
"""
import os
cd=os.path.dirname(__file__)
import sys
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib as mpl
import yaml
import re
import glob
import warnings
warnings.filterwarnings('ignore')

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.size'] = 12
mpl.rcParams['savefig.dpi']=300

plt.close('all')

#%% Inputs
if len(sys.argv)==1:
    source_config=os.path.join(cd,'configs/config.yaml')
    sdate='2023-12-05T02:00:00'
    edate='2023-12-05T05:00:00'
else:
    source_config=sys.argv[1]
    sdate=sys.argv[2]
    edate=sys.argv[3]
    
#fixed inputs
file_duration=600#[s] scan file duration
scan_duration=20#[s] scan duration
max_time_diff=600#[s] maximum time difference lidar - scada
max_dep=3#max deprojection factor
H=90#[m] hub height

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

#file search
files={}
times={}
for s in config['source_rhi']:
    files_all=np.array(sorted(glob.glob(config['source_rhi'][s])))
    times_all=dates_from_files(files_all)
    sel=(times_all>np.datetime64(sdate))*(times_all<np.datetime64(edate)+np.timedelta64(file_duration,'s'))
    
    files[s]=files_all[sel]
    times[s]=times_all[sel]
    
#layout
Turbines=xr.open_dataset(config['source_layout'],group='turbines')
x_T={}
y_T={}
for s in config['source_rhi']:
    x_T[s]=Turbines['x_utm'][Turbines['name']==config['turbine_names'][s]].values[0]
    y_T[s]=Turbines['y_utm'][Turbines['name']==config['turbine_names'][s]].values[0]
    
#%% Main
Data_all={}
time_avg_all=np.array([],dtype='datetime64')
for s in config['source_rhi']:
    stack=[]
    for f in files[s]:
        Data=xr.open_dataset(f)
        
        time_avg=Data.time.isel(beamID=0)+(Data.time.isel(beamID=-1)-Data.time.isel(beamID=0))/2
        Data=Data.rename({'scanID':'time_avg'})
        Data=Data.assign_coords(time_avg=('time_avg', time_avg.values))

        stack=stack+[Data[['wind_speed','x','y','z']].where(Data.qc_wind_speed==0)]
    
    if len(stack)>0:
        Data_all[s]=xr.concat(stack, dim='time_avg')
        time_avg_all=np.append(time_avg_all,Data_all[s].time_avg.values)
        
        
time_bins=np.arange(time_avg_all.min()-np.timedelta64(scan_duration,'s')/2,time_avg_all.max()+np.timedelta64(scan_duration+1,'s')/2,np.timedelta64(scan_duration,'s'))

#load scada
yaw={}
for s in Data_all.keys():
    files=glob.glob(config['source_scada'][s])
    if len(files)>0:
        Data=xr.open_mfdataset(files).rename({'WTUR.DateTime':'time'})
        
        tnum=(Data_all[s].time_avg -np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
        tnum_scd=(Data.time -np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
        
        time_diff=tnum_scd.interp(time=Data_all[s].time_avg,method="nearest")-tnum
        cos=np.cos(np.radians(Data['WNAC.Dir_10m_Avg'])).interp(time=Data_all[s].time_avg)
        sin=np.sin(np.radians(Data['WNAC.Dir_10m_Avg'])).interp(time=Data_all[s].time_avg)
        Data_all[s]['yaw']=(np.degrees(np.arctan2(sin,cos)).where(np.abs(time_diff)<max_time_diff)%360).drop_vars("time")
    
os.makedirs(os.path.join(cd,'figures/rhi_video'),exist_ok=True)
for t1,t2 in zip(time_bins[:-1],time_bins[1:]):
    found=0
    for s in Data_all.keys():
        sel=(Data_all[s].time_avg>t1)*(Data_all[s].time_avg<t2)
        if np.sum(sel)>0:
            sel=np.where(sel)[0][0]
            found+=1
            
            if found==1:
                fig=plt.figure(figsize=(18,18))
                ax1=fig.add_subplot(2,1,1)
                ax2=fig.add_subplot(2,1,2,projection='3d')
                
            Data_sel=Data_all[s].isel(time_avg=sel)
            x=(Data_sel.x*np.cos(np.radians(270-Data_sel.yaw))-Data_sel.y*np.sin(np.radians(270-Data_sel.yaw))).values.ravel()
            y=(Data_sel.x*np.sin(np.radians(270-Data_sel.yaw))+Data_sel.y*np.cos(np.radians(270-Data_sel.yaw))).values.ravel()
            z=Data_sel.z.values.ravel()
            dep=1/(Data_sel.x/(Data_sel.x**2+Data_sel.z**2)**0.5)
            dep=dep.where(np.abs(dep)<max_dep)
            u=(Data_sel.wind_speed*dep).values.ravel()
            
            sc1=ax1.scatter(y+y_T[s],z+H,s=4,c=u,cmap='coolwarm',vmin=0,vmax=20)
            ax1.plot(y_T[s],H,'.k',markersize=20)
            ax1.set_xlim([-2000+y_T['rt1'],8000+y_T['rt1']])
            ax1.set_ylim([0,2000])
            ax1.set_aspect('equal')
            ax1.set_xlabel('S-N [m]')
            ax1.set_ylabel('$z$ [m a.g.l.]')
            ax1.grid(True)
            
            sc2=ax2.scatter(x+x_T[s],y+y_T[s],z+H,s=2,c=u,cmap='coolwarm',vmin=0,vmax=20)
            ax2.quiver(x_T[s],y_T[s], 0, np.cos(np.radians(270-Data_sel.yaw))*500, np.sin(np.radians(270-Data_sel.yaw))*500, 0, color='k', arrow_length_ratio=0.25)
            ax2.set_xlim([-2000+x_T['rt1'],2000+x_T['rt1']])
            ax2.set_ylim([-2000+y_T['rt1'],8000+y_T['rt1']])
            
            ax2.set_aspect('equal')
            ax2.set_xlabel('W-E [m]')
            ax2.set_ylabel('S-N [m]')
            ax2.set_zlabel('$z$ [m a.g.l.]')
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.set_zticklabels([])
            #ax2.view_init([-60,30])
           
    if found>0:
        ax1.set_title(str(t1+(t2-t1)/2).replace('T',' ')[:-10])
            
        plt.colorbar(sc1,label=r'Deprojected wind speed [m s$^{-1}$]')
        plt.subplots_adjust(top=0.95,
                        bottom=0.05,
                        left=0.05,
                        right=0.95,
                        hspace=0.0,
                        wspace=0.0)
        plt.savefig(os.path.join(cd,'figures/rhi_video',f'{str(t1)[:-10].replace("T",".").replace("-","").replace(":","")}.png'))
            
        plt.close()
        print(t1)
        
    
            