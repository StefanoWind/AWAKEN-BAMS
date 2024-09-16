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
import re
from matplotlib.patches import Circle

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12

#%% Inputs
source_config=os.path.join(cd,'config.yaml')
source_log='data/20230101.000500-20240101.224500.awaken.glob.summary.csv'

#dataset
channel='awaken/rt1.lidar.z02.a0'
regex='\d{8}\.\d{1}\d*[13579]20\d{2}\.user5.nc'#regexp to select data from DAP
sdate='20230501000000'#start date
edate='20230901000000'#end date

#stats
max_TI=50#[%] maximum TI
WS_bin=[5,8]#[m/s] bin in wind speed
WD_bin=[135,225]#[deg] bin in wind direction
TI_bin=[0,10]#[%] bins i turbulent intensity
perc_lim=[5,95]#[%] outlier rejection
p_value=0.05#p-value for confidence interval
max_err_u=0.2#maximumm error in mean normalized streamwise velocity
max_err_TI=4#maximumm error in mean turbulence intensity

#site
D=127#[m] rotor diameter
H=90#[m] hub height

#graphics
#plot area[D]
ymin=-2
ymax=2
zmin=-1
zmax=2
        
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
LOG=LOG.set_index('UTC Time')

#WDH setup
_filter = {
    'Dataset': channel,
    'date_time': {
        'between': [sdate,edate]
    },
    'file_type': 'nc',
    'ext1':'user5', 
}
a2e = DAP('a2e.energy.gov',confirm_downloads=False)

#naming
dirname=(str(WS_bin)+str(WD_bin)+str(TI_bin)).replace(' ','').replace(',','.').replace('[','.').replace(']','')[1:]

#%% Main

#qc log
LOG['Rotor-averaged TI [%]']=LOG['Rotor-averaged TI [%]'].where(LOG['Rotor-averaged TI [%]']>0).where(LOG['Rotor-averaged TI [%]']<max_TI)

#list files
files_dap=a2e.search(_filter)
if files_dap is None:
    a2e.setup_two_factor_auth(username=config['username'], password=config['password'])
    files_dap=a2e.search(_filter)

#select files
tnum_file=[utl.datenum(f['date_time'],'%Y%m%d%H%M%S') for f in files_dap]

WS_file=   np.interp(tnum_file,tnum_log,LOG['Hub-height wind speed [m/s]'])
c=np.interp(tnum_file,tnum_log,utl.cosd(LOG['Hub-height wind direction [degrees]']))
s=np.interp(tnum_file,tnum_log,utl.sind(LOG['Hub-height wind direction [degrees]']))
WD_file=utl.cart2pol(c,s)[1]
TI_file=   np.interp(tnum_file,tnum_log,LOG['Rotor-averaged TI [%]'])

sel_vol=np.array([len(re.findall(regex, f['Filename']))>0 for f in files_dap])
sel_WS=(WS_file>=WS_bin[0])*(WS_file<WS_bin[1])
sel_WD=(WD_file>=WD_bin[0])*(WD_file<WD_bin[1])
sel_TI=(TI_file>=TI_bin[0])*(TI_file<TI_bin[1])
files_dap_sel=np.array(files_dap)[sel_WS*sel_WD*sel_TI*sel_vol]

#download selected files
a2e.download_files(list(files_dap_sel),os.path.join(config['path_data'],channel,dirname),replace=False)

#data standardization and LiSBOA
files=glob.glob(os.path.join(config['path_data'],channel,dirname,'*nc'))
for f in files:
    try:
        lproc_b0 = lb0.LIDARGO(f,os.path.join(cd,config['path_config_b0']), verbose=True)
        lproc_b0.process_scan(f,replace=False,save_file=True)
        
        lproc_c0 = lc0.LIDARGO(lproc_b0.save_filename,os.path.join(cd,config['path_config_c0']), verbose=True)
        if 'config' in dir(lproc_c0):
            lproc_c0.process_scan(lproc_b0.save_filename,replace=False,save_file=True)
            
    except Exception as e:
        print(e)

#concatenate statistics
files=glob.glob(os.path.join(config['path_data'],channel.replace('a0','c0'),dirname,'*nc'))
u_all=[]
TI_all=[]
ctr=0
for f in files:
    Data=xr.open_dataset(f)
    WS_inf=WS_file[sel_WS*sel_WD*sel_TI*sel_vol][ctr]
    u=Data['u_avg']/WS_inf
    TI=Data['u_stdev']/Data['u_avg']*100
    u_all.append(u.values)
    TI_all.append(TI.values)
    ctr+=1
    
u_all=np.array(u_all)  
TI_all=np.array(TI_all)  
        
#global average
u_avg=xr.DataArray(np.apply_along_axis(lambda x:utl.filt_mean(x,perc_lim=perc_lim), axis=0,                             arr=u_all),coords=Data.coords)
u_low=xr.DataArray(np.apply_along_axis(lambda x:utl.filt_BS_mean(x,perc_lim=perc_lim,p_value=p_value/2*100), axis=0,    arr=u_all),coords=Data.coords)
u_top=xr.DataArray(np.apply_along_axis(lambda x:utl.filt_BS_mean(x,perc_lim=perc_lim,p_value=(1-p_value/2)*100), axis=0,arr=u_all),coords=Data.coords)
u_avg_qc=u_avg.where(u_top-u_low<max_err_u)

TI_avg=xr.DataArray(np.apply_along_axis(lambda x:utl.filt_mean(x,perc_lim=perc_lim), axis=0,                             arr=TI_all),coords=Data.coords)
TI_low=xr.DataArray(np.apply_along_axis(lambda x:utl.filt_BS_mean(x,perc_lim=perc_lim,p_value=p_value/2*100), axis=0,    arr=TI_all),coords=Data.coords)
TI_top=xr.DataArray(np.apply_along_axis(lambda x:utl.filt_BS_mean(x,perc_lim=perc_lim,p_value=(1-p_value/2)*100), axis=0,arr=TI_all),coords=Data.coords)
TI_avg_qc=TI_avg.where(TI_top-TI_low<max_err_TI)

#%% Output
Output=xr.Dataset()
Output['u_avg']=u_avg
Output['TI_avg']=TI_avg
Output['files']=xr.DataArray(data=[f['Filename'] for f in files_dap_sel],coords={'index':np.arange(len(files_dap_sel))})
Output['WS']=xr.DataArray(data=WS_file[sel_WS*sel_WD*sel_TI*sel_vol],coords={'index':np.arange(len(files_dap_sel))})
Output['WD']=xr.DataArray(data=WD_file[sel_WS*sel_WD*sel_TI*sel_vol],coords={'index':np.arange(len(files_dap_sel))})
Output['TI']=xr.DataArray(data=TI_file[sel_WS*sel_WD*sel_TI*sel_vol],coords={'index':np.arange(len(files_dap_sel))})
Output.attrs['start_date']=sdate
Output.attrs['end_date']=edate
Output.to_netcdf(os.path.join(config['path_data'],channel,dirname,dirname+'.stats.nc'))

#%% Plots
plt.close('all')
x_plot_wake=[3,5,7,10,12]
D=127
x=Data['x'].values
y=Data['y'].values
z=Data['z'].values
fig=plt.figure(figsize=(12,10))
ctr=1
for x_plot in x_plot_wake:
    u_avg_int=u_avg_qc.interp(x=x_plot*D,method='linear').values
    TI_avg_int=TI_avg_qc.interp(x=x_plot*D,method='linear').values
    
    #Plot mean velocity
    ax = plt.subplot(len(x_plot_wake),2,(ctr-1)*2+1)
    if ctr==1:
        plt.title('Mean streamwise velocity: \n'+r'$U_\infty\in['+str(WS_bin[0])+','+str(WD_bin[1])+r')$ m s$^{-1}$, $\theta_w\in['+str(WD_bin[0])+','+str(WS_bin[1])+r')^\circ$, TI$\in['+str(TI_bin[0])+','+str(TI_bin[1])+')\%$'+'\n $x='+str(x_plot)+'D$')
    else:
        plt.title(r'$x='+str(x_plot)+'D$')
            
    ax.set_facecolor((0,0,0,0.2))
    cf1=plt.contourf(y,z,u_avg_int.T,np.round(np.linspace(np.nanpercentile(u_avg,5), np.nanpercentile(u_avg,95), 10),2), cmap='coolwarm',extend='both')
    plt.grid(alpha=0.5)
    

    circle = Circle((0, 0), D/2, edgecolor='k', facecolor='none')
    ax.add_patch(circle)
    plt.plot(y,y*0-H,'k')
    
    plt.xlabel(r'$y$ [m]') 
    plt.ylabel(r'$z$ [m]')
    
    plt.xlim([ymin*D,ymax*D])
    plt.ylim([zmin*D,zmax*D])
    
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    ax.set_box_aspect(np.diff(ylim)/np.diff(xlim))
        
    ax = plt.subplot(len(x_plot_wake),2,(ctr-1)*2+2)
    ax.set_facecolor((0,0,0,0.2))
    if ctr==1:
        plt.title('Turbulence intensity: \n'+r'$U_\infty\in['+str(WS_bin[0])+','+str(WS_bin[1])+r')$ m s$^{-1}$, $\theta_w\in['+str(WD_bin[0])+','+str(WD_bin[1])+r')^\circ$, TI$\in['+str(TI_bin[0])+','+str(TI_bin[1])+')\%$'+'\n $x='+str(x_plot)+'D$')
    else:
        plt.title(r'$x='+str(x_plot)+'D$')
        
    cf2=plt.contourf(y,z,TI_avg_int.T,np.unique(np.round(np.linspace(np.nanpercentile(TI_avg,5)-0.5, np.nanpercentile(TI_avg,95)+0.5, 20),1)), cmap='hot',extend='both')
    plt.grid(alpha=0.5)
    
    circle = Circle((0, 0), D/2, edgecolor='k', facecolor='none')
    ax.add_patch(circle)
    plt.plot(y,y*0-H,'k')
    
    plt.xlabel(r'$y$ [m]') 
    plt.ylabel(r'$z$ [m]')
    
    plt.xlim([ymin*D,ymax*D])
    plt.ylim([zmin*D,zmax*D])
    
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    ax.set_box_aspect(np.diff(ylim)/np.diff(xlim))
    ctr+=1
utl.remove_labels(fig)

axs=fig.axes
cax = fig.add_axes([axs[-2].get_position().x0+axs[-2].get_position().width+0.0075,axs[-2].get_position().y0,
                    0.01,axs[0].get_position().y0+axs[0].get_position().height-axs[-2].get_position().y0])
plt.colorbar(cf1,cax=cax,label=r'Mean streamwise velocity [m s$^{-1}$]')
cax = fig.add_axes([axs[-1].get_position().x0+axs[-1].get_position().width+0.0075,axs[-1].get_position().y0,
                    0.01,axs[1].get_position().y0+axs[1].get_position().height-axs[-1].get_position().y0])
plt.colorbar(cf2,cax=cax,label=r'Turbulence intensity [%]')

plt.savefig(os.path.join(config['path_data'],channel,dirname,dirname+'.stats.png'))