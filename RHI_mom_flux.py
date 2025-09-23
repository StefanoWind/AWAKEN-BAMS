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
from matplotlib.gridspec import GridSpec
from lisboa import statistics as stats
import matplotlib
import scipy as sp
from scipy.optimize import curve_fit
import yaml
import re
import glob
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['savefig.dpi'] = 500
plt.close('all')

#%% Inputs
if len(sys.argv)==1:
    source_config=os.path.join(cd,'configs/config.yaml')
    ws_lim=[10.0,30.0]#[m/s] LLJ nose wind speed range
    wd_lim=180.0#[deg] max misalignment
    ti_lim=[0.0,10.0]#[%] TI range
    llj_lim=[300.0,500.0]#[m] LLJ nose height limits
else:
    source_config=sys.argv[1]
    ws_lim=[np.float64(sys.argv[2]),np.float64(sys.argv[3])]
    wd_lim=np.float64(sys.argv[4])
    ti_lim=[np.float64(sys.argv[5]),np.float64(sys.argv[6])]
    llj_lim=[np.float64(sys.argv[7]),np.float64(sys.argv[8])]

#fixed inputs
source_log=os.path.join(cd,'data/glob.lidar.eventlog.avg.c2.20230101.000500.csv')#inflow table source
wd_ref=180#[deg] aligned wind directipon
min_cos=1/3 #minimum cosine for de-projection
scan_duration=600#[s] scan duration
inflow_site='A1' 
outflow_site='H'
H=90#[m] hub height
z_hub=110#[m] selected height for hub hight conditions
D=127 #[m] diameter
# ele_corr=2#[deg] elevatin correction

#stats
perc_lim=[5,95]#[%] percentile limits
min_u=0.1 #minimum normalized wind speed
max_u=1.5#maximum normalized wind speed
min_du=-0.5 #minimum normalized wind speed difference
max_du=0.5 #maximum normalized wind speed difference
dz=200
z_max=500

config_lisboa={'sigma':0.25,
        'mins':[-1800,0],
        'maxs':[8250,1000],
        'Dn0':[127*4,127],
        'r_max':3,
        'dist_edge':1,
        'tol_dist':0.1,
        'grid_factor':0.25,
        'max_Dd':1,
        'max_iter':3}

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

def radial_wind_speed(ele,U,ele0):
    return -U*np.cos(np.radians(ele+ele0))

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
du=[]
WS_inflow=[]
WS_outflow=[]
uw_inflow=[]
uw_outflow=[]

#file system
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
                
                Data=xr.open_dataset(f)
                if len(Data.beamID)==1:
                    print(f'Invalid file {f}')
                    continue
                time_avg=(Data.time.isel(scanID=0,beamID=0)+(Data.time.isel(scanID=-1,beamID=-1)-Data.time.isel(scanID=0,beamID=0))/2).values
                Data=Data.where(Data.qc_wind_speed==0)
   
                #bias correction
                Data['wind_speed']=Data.wind_speed-config['bias'][s]
                    
                #tilt correction
                Data['elevation']=Data.elevation.transpose('range','beamID','scanID')
                rws_sel=(Data.wind_speed.where(Data.z>z_max-dz)).values.ravel()
                ele_sel=(Data.elevation.where(Data.z>z_max-dz)).values.ravel()
                real=~np.isnan(rws_sel+ele_sel)
                popt, pcov = curve_fit(radial_wind_speed, ele_sel[real],rws_sel[real], p0=[10,0])
                print(f'Optimal elevation = {popt[1]} deg')
                
                Data['elevation']=Data.elevation+Data.pitch
                Data['x_corr']=Data.range*np.cos(np.radians(Data.elevation))*np.cos(np.radians(90-Data.azimuth))
                Data['z_corr']=Data.range*np.sin(np.radians(Data.elevation))+H
                
                Data=Data.where(np.abs(np.cos(np.radians(Data.elevation)))>min_cos)
                
                #exclude upstream wake
                if s!='rt1':
                    Data=Data.where((Data.x_corr>0)+(Data.z_corr>H+D/2))
                
                real=~np.isnan(Data.x_corr+Data.z_corr+Data.wind_speed).values
                x=np.append(x,Data.x_corr.values[real]+config['turbine_x'][s])
                z=np.append(z,Data.z_corr.values[real])
                
                #inflow
                date=os.path.basename(f).split('.')[4]
                file_inflow=glob.glob(os.path.join(config['source_prof'][inflow_site],f'*{date}*nc'))
                if len(file_inflow)==1:
                    Data_inflow=xr.open_dataset(file_inflow[0])
                    WS=Data_inflow.WS.interp(time=[time_avg]).squeeze()
                    cos=np.cos(np.radians(Data_inflow.WD)).interp(time=[time_avg]).squeeze()
                    sin=np.sin(np.radians(Data_inflow.WD)).interp(time=[time_avg]).squeeze()
                    WD=np.degrees(np.arctan2(sin,cos))%360
                else:
                    continue
                    
                U_inf=(ws_int1[files==f]+ws_int2[files==f])/2
                cos=np.cos(np.radians(WD.sel(height=slice(H-D/2,H+D/2)).mean().values))
                sin=np.sin(np.radians(WD.sel(height=slice(H-D/2,H+D/2)).mean().values))
                WD_hub=np.degrees(np.arctan2(sin,cos))%360
                
                WS_int=np.interp(Data.z_corr.values,WS.height.values,WS.values)
                Data['WS']=xr.DataArray(WS_int,coords=Data.z_corr.coords)
                
                cos=np.interp(Data.z_corr.values,WD.height.values,np.cos(np.radians(WD.values-WD_hub)))
                sin=np.interp(Data.z_corr.values,WD.height.values,np.sin(np.radians(WD.values-WD_hub)))
                Data['dWD']=xr.DataArray(np.degrees(np.arctan2(sin,cos))%360,coords=Data.z_corr.coords)
                
                u_eq=-Data.wind_speed/np.cos(np.radians(Data.elevation))/np.cos(np.radians(Data.dWD))/U_inf
                du_eq=u_eq-Data['WS']/U_inf
                
                u=np.append(u,u_eq.values[real])
                du=np.append(du,du_eq.values[real])
                
                #mom flux
                date=os.path.basename(f).split('.')[4]
                file_inflow=glob.glob(os.path.join(config['source_prof'][inflow_site],f'*{date}*nc'))
                if len(file_inflow)==1:
                    Data_inflow=xr.open_dataset(file_inflow[0])
                    WS_inflow_int=Data_inflow.WS.interp(time=[time_avg]).squeeze()/U_inf
                    uw_inflow_int=Data_inflow.uw.interp(time=[time_avg]).squeeze()/U_inf**2
                    if len(uw_inflow)==0:
                        WS_inflow=WS_inflow_int.values
                        uw_inflow=uw_inflow_int.values
                    else:
                        WS_inflow=np.vstack([WS_inflow,WS_inflow_int.values])
                        uw_inflow=np.vstack([uw_inflow,uw_inflow_int.values])
                    
                file_outflow=glob.glob(os.path.join(config['source_prof'][outflow_site],f'*{date}*nc'))
                if len(file_outflow)==1:
                    Data_outflow=xr.open_dataset(file_outflow[0])
                    WS_outflow_int=Data_outflow.WS.interp(time=[time_avg]).squeeze()/U_inf
                    uw_outflow_int=Data_outflow.uw.interp(time=[time_avg]).squeeze()/U_inf**2
                    if len(uw_outflow)==0:
                        WS_outflow=WS_outflow_int.values
                        uw_outflow=uw_outflow_int.values
                    else:
                        WS_outflow=np.vstack([WS_outflow,WS_outflow_int.values])
                        uw_outflow=np.vstack([uw_outflow,uw_outflow_int.values])
                        
                print(f'{f} done',flush=True)
                
                #plots
                plt.figure(figsize=(18,8))
                ax=plt.subplot(2,1,1)
                plt.scatter(Data.x_corr.values[real]+config['turbine_x'][s],Data.z_corr.values[real],s=1,c=u_eq.values[real],cmap='coolwarm',vmin=0.25,vmax=1)
                plt.plot(config['inflow_x']+uw_inflow_int.height*0,uw_inflow_int.height,'--k')
                plt.plot(config['inflow_x']+uw_inflow_int*1000000,uw_inflow_int.height,'g')
                plt.plot(config['outflow_x']+uw_outflow_int.height*0,uw_outflow_int.height,'--k')
                plt.plot(config['outflow_x']+uw_outflow_int*1000000,uw_outflow_int.height,'g')
                plt.title(os.path.basename(f))
                ax=plt.gca()
                ax.set_aspect('equal')
                plt.xlim([-2000,8500])
                plt.ylim([0,1250])
                plt.xlabel(r'$x$ [m]')
                plt.ylabel(r'$y$ [m]')
                plt.grid()
                plt.colorbar(label=r'$u/U_\infty$')
                
                ax=plt.subplot(2,1,2)
                plt.scatter(Data.x_corr.values[real]+config['turbine_x'][s],Data.z_corr.values[real],s=1,c=du_eq.values[real],cmap='seismic',vmin=-0.25,vmax=0.25)
                ax=plt.gca()
                ax.set_aspect('equal')
                plt.xlim([-2000,8500])
                plt.ylim([0,1250])
                plt.xlabel(r'$x$ [m]')
                plt.ylabel(r'$y$ [m]')
                plt.grid()
                plt.colorbar(label=r'$\Delta u/U_\infty$')
                
                plt.savefig(os.path.join(cd,'figures',os.path.basename(save_name)[:-3],os.path.basename(f).replace('nc','png')))
                plt.close()
                
    #output
    Output=xr.Dataset()
    Output['x']=xr.DataArray(x,coords={'index':np.arange(len(x))})
    Output['z']=xr.DataArray(z,coords={'index':np.arange(len(x))})
    Output['u']=xr.DataArray(u,coords={'index':np.arange(len(x))})
    Output['du']=xr.DataArray(du,coords={'index':np.arange(len(x))})
    Output['WS_inflow']= xr.DataArray(WS_inflow, coords={'index2':np.arange(len(WS_inflow[:,0])), 'height':WS_inflow_int.height})
    Output['WS_outflow']=xr.DataArray(WS_outflow,coords={'index2':np.arange(len(WS_outflow[:,0])),'height':WS_outflow_int.height})
    Output['uw_inflow']= xr.DataArray(uw_inflow, coords={'index2':np.arange(len(uw_inflow[:,0])), 'height':uw_inflow_int.height})
    Output['uw_outflow']=xr.DataArray(uw_outflow,coords={'index2':np.arange(len(uw_outflow[:,0])),'height':uw_outflow_int.height})
    Output.to_netcdf(save_name)
    Output.close()

#load data
Data=xr.open_dataset(save_name)

#uw stats
WS_inflow_avg=  np.apply_along_axis(lambda x: utl.filt_stat(x,   np.nanmean,perc_lim=perc_lim), axis=0, arr=Data.WS_inflow.values)
WS_outflow_avg= np.apply_along_axis(lambda x: utl.filt_stat(x,   np.nanmean,perc_lim=perc_lim), axis=0, arr=Data.WS_outflow.values)

uw_inflow_avg=  np.apply_along_axis(lambda x: utl.filt_stat(x,   np.nanmean,perc_lim=perc_lim), axis=0, arr=Data.uw_inflow.values)
uw_outflow_avg= np.apply_along_axis(lambda x: utl.filt_stat(x,   np.nanmean,perc_lim=perc_lim), axis=0, arr=Data.uw_outflow.values)

#lisboa
x_exp=[Data.x.values.ravel(),Data.z.values.ravel()]
lproc=stats.statistics(config_lisboa)  

f=Data.u.where((Data.u>=min_u)*(Data.u<=max_u)).values.ravel()
grid,Dd,excl,u_avg,hom=lproc.calculate_statistics(x_exp,f)
x_grid=grid[0]
z_grid=grid[1]
u_avg[excl]=np.nan

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

x_exp=[Data.x.values.ravel(),Data.z.values.ravel()]
lproc=stats.statistics(config_lisboa) 
f=Data.du.where((Data.du>=min_du)*(Data.du<=max_du)).values.ravel()
grid,Dd,excl,du_avg,hom=lproc.calculate_statistics(x_exp,f)
du_avg[excl]=np.nan

#inpainting
interp_limit = 5
valid_mask1 = ~np.isnan(du_avg)
distance1 = sp.ndimage.distance_transform_edt(~valid_mask1)
interp_mask1 = (np.isnan(du_avg)) & (distance1 <= interp_limit)
yy1, xx1 = np.indices(du_avg.shape)
points1 = np.column_stack((yy1[valid_mask1], xx1[valid_mask1]))
values1 = du_avg[valid_mask1]
interp_points1 = np.column_stack((yy1[interp_mask1], xx1[interp_mask1]))
interpolated_values1 = sp.interpolate.griddata(points1, values1, interp_points1, method='linear')
du_avg_inp = du_avg.copy()
du_avg_inp[interp_mask1] = interpolated_values1

#LLJ height
LLJ_nose=z_grid[np.nanargmax(u_avg_inp,axis=1)]

#%% Plots
plt.close('all')
skip=int(np.ceil(len(Data.x)/max_plot))
plt.figure(figsize=(18,4))
plt.scatter(Data.x.values[::skip],Data.z.values[::skip],s=1,c=Data.u.values[::skip],cmap='coolwarm',vmin=0.5,vmax=0.95)
ax=plt.gca()
ax.set_aspect('equal')
plt.xlim([-1800,7800])
plt.ylim([0,1000])
plt.grid()

fig=plt.figure(figsize=(18,5))
matplotlib.rcParams['savefig.dpi'] = 500
gs = GridSpec(nrows=2, ncols=3, width_ratios=[1,6,0.25], figure=fig)

ax=fig.add_subplot(gs[0,0])
plt.plot(Data.height*0,Data.height,'--k')
plt.plot(WS_inflow_avg,Data.height,'-g',label='Inflow')
plt.plot(WS_outflow_avg,Data.height,'-m',label='Outflow')
plt.ylim([0,1000])
plt.xlim([0,1.5])
plt.ylabel(r'$z$ [m a.g.l.]')
plt.xlabel(r'$U/U_\infty^2$')
plt.grid()
plt.legend(draggable=True)

ax=fig.add_subplot(gs[0,1])
cf=plt.contourf(x_grid,z_grid,u_avg_inp.T,np.arange(0.4,1.01,0.05),cmap='coolwarm',extend='both')
plt.contour(x_grid,z_grid,u_avg_inp.T,np.arange(0.4,1.01,0.05),extend='both',linewidths=1,alpha=0.25,colors='k')
for s in config['source_rhi']:
    plt.plot([config['turbine_x'][s],config['turbine_x'][s]],[-D/2+H,D/2+H],'k',linewidth=1)
plt.plot(x_grid,LLJ_nose,'.k',markersize=10,markerfacecolor='w')
ax=plt.gca()
ax.set_aspect('equal')
ax.set_yticklabels([])
plt.xlim([-1800,8100])
plt.ylim([0,1000])
plt.grid()
plt.xlabel(r'$x$ [m]')
plt.grid()

plt.plot([config['inflow_x'],config['inflow_x']],[0,1000],'--g',linewidth=2)
plt.plot([config['outflow_x'],config['outflow_x']],[0,1000],'--m',linewidth=2)

cax=fig.add_subplot(gs[0,2])
plt.colorbar(cf,cax=cax,label=r'$\overline{u}/U_\infty$ [m s$^{-1}$]')

ax=fig.add_subplot(gs[1,0])
plt.plot(Data.height*0,Data.height,'--k')
plt.plot(uw_inflow_avg,Data.height,'-g',label='Inflow')
plt.plot(uw_outflow_avg,Data.height,'-m',label='Outflow')
plt.ylim([0,1000])
plt.xlim([-0.0001,0.00001])
plt.xticks([-0.0001,0],labels=[r'$-10^{-4}$',r'$0$'])
plt.ylabel(r'$z$ [m a.g.l.]')
plt.xlabel(r'$\overline{u^\prime w^\prime}/U_\infty^2$')
plt.grid()

ax=fig.add_subplot(gs[1,1])
cf=plt.contourf(x_grid,z_grid,du_avg.T,np.arange(-0.25,0.25,0.01),cmap='seismic',extend='both')
plt.contour(x_grid,z_grid,du_avg_inp.T,np.arange(-0.25,0.25,0.01),extend='both',linewidths=1,alpha=0.25,colors='k')
for s in config['source_rhi']:
    plt.plot([config['turbine_x'][s],config['turbine_x'][s]],[-D/2+H,D/2+H],'k',linewidth=1)
plt.plot(x_grid,LLJ_nose,'.k',markersize=10,markerfacecolor='w')
ax=plt.gca()
ax.set_aspect('equal')
ax.set_yticklabels([])
plt.xlim([-1800,8100])
plt.ylim([0,1000])
plt.grid()
plt.xlabel(r'$x$ [m]')
plt.grid()

plt.plot([config['inflow_x'],config['inflow_x']],[0,1000],'--g',linewidth=2)
plt.plot([config['outflow_x'],config['outflow_x']],[0,1000],'--m',linewidth=2)

cax=fig.add_subplot(gs[1,2])
plt.colorbar(cf,cax=cax,label=r'$\Delta\overline{u}/U_\infty$ [m s$^{-1}$]',ticks=[-0.2,-0.1,0,0.1,0.2])
plt.tight_layout()

