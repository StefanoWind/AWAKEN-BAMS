# -*- coding: utf-8 -*-
"""
Daily statisitcs from ASSISTs+TROPoe
"""
import os
cd=os.path.dirname(__file__)
import numpy as np
import utils as utl
import yaml
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import stats
import glob

mpl.rcParams.update({
"savefig.format": "png",
"savefig.dpi":500,
"pdf.fonttype": 42,
"ps.fonttype": 42,
"font.family": "serif",
"font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
"mathtext.fontset": "custom",
"mathtext.rm": "serif",
"mathtext.it": "serif:italic",
"mathtext.bf": "serif:bold",
"axes.labelsize": 16,
"axes.titlesize": 16,
"xtick.labelsize": 14,
"ytick.labelsize": 14,
"legend.fontsize": 14,
"lines.linewidth": 1,
"lines.markersize": 4,
})
plt.close('all')

#%% Inputs
source_config=os.path.join(cd,'configs/config.yaml')

#dataset
site='C1a'#selected site
#TROPoe data paths
sources_trp={'B': 'C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/awaken_processing/data/awaken/sb.assist.tropoe.z01.c0/*nc',
             'C1a':'C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/awaken_processing/data/awaken/sc1.assist.tropoe.z01.c0/*nc',
             'G': 'C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/awaken_processing/data/awaken/sg.assist.tropoe.z01.c0/*nc'}
H=90#[m] hub height
D=127#[m] diameter

#qc
min_lwp=5#[g/cm^2] minimum liquid water path to flag clouds
max_gamma=1#maximum convergence factor
max_rmsa=5#maximum RMS of retrieval

#stats
max_height=2000#[m] maixmum height
bin_hour=np.arange(25)# bin in hour
perc_lim=[5,95]#[%] outlier rejection
p_value=0.05#p-value for confidence interval
max_err={'temperature':4,'waterVapor':2,'dtheta_v_dz':0.01,'pblh':0.1}#maximum 95% c.i. width

#user
variables=['temperature','waterVapor','dtheta_v_dz','pblh']

#graphics
labels={'temperature':r'$T$ [$^\circ$C]','waterVapor':r'$r$ [g Kg$^{-1}$]','dtheta_v_dz':r'$\partial{\theta_v}~\partial z^{-1}$ [$^\circ$C m$^{-1}$]'}
ticks={'temperature':np.arange(10,22.1,0.5),'waterVapor':np.arange(4,9.1,0.25),'dtheta_v_dz':np.arange(-0.1,0.11,0.005)}
colormaps={'temperature':'coolwarm','waterVapor':'Blues','dtheta_v_dz':'seismic'}

#%% Initialization
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)

#read and QC all TROPoe data
save_path=os.path.join(cd,'data',f'TROPoe_{site}.nc')
if not os.path.isfile(save_path):
   
    files=glob.glob(sources_trp[site])
    Data=xr.open_mfdataset(files).sel(height=slice(0,(max_height+100)/1000))
    
    #qc tropoe data
    Data['cbh'][(Data['lwp']<min_lwp).compute()]=Data['height'].max()+1000#remove clouds with low lwp

    qc_gamma=Data['gamma']<=max_gamma
    qc_rmsa=Data['rmsa']<=max_rmsa
    qc_cbh=Data['height']<Data['cbh']
    qc=qc_gamma*qc_rmsa*qc_cbh
    Data['qc']=~qc+0
    
    Data=Data.where(Data.qc==0)
    print(f"{int(np.sum(Data.qc!=0))} points fail QC in TROPoe at site {site}")
    
    #fix height
    Data=Data.assign_coords(height=Data.height*1000)
    
    raise BaseException()
    #add lapse rate
    q=Data['waterVapor']/1000/(1+Data['waterVapor']/1000)
    Data['theta_v']=Data['theta']*(1+0.61*q)
    Data['dtheta_v_dz']=Data['theta_v'].differentiate("height")
    
    Data[variables].to_netcdf(save_path)
    Data.close()

Data=xr.open_dataset(save_path)

Data['pblh']=Data.pblh.where(Data.pblh>0.3)

#data extraction
height=Data.height.values
tnum=np.float64(Data.time)/10**9
hour=np.mod(tnum,3600*24)/3600#time of the day
hour_avg=utl.mid(bin_hour)

#zeroing
Data_avg={}

#%% Main

#daily average
Data_avg=xr.Dataset()
for v in variables:
    f_avg_all=[]
    for h in height:
        f=Data[v].sel(height=h).values
        f[f==0]=np.nan
        real=~np.isnan(f)
        
        if np.sum(real)>0:
            f_avg= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_stat(x,np.nanmean,perc_lim=perc_lim),                             bins=bin_hour)[0]
            f_low= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100),    bins=bin_hour)[0]
            f_top= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=bin_hour)[0]

            f_avg[np.abs(f_top-f_low)>max_err[v]]=np.nan
            f_avg[np.isnan(np.abs(f_top-f_low))]=np.nan
            
            f_avg_all=utl.vstack(f_avg_all,f_avg)

        else:
            f_avg_all=utl.vstack(f_avg_all,hour_avg*np.nan)
       
    Data_avg[v]=xr.DataArray(data=f_avg_all.T,coords={'hour':hour_avg,'height':Data.height.values}).interpolate_na(dim='height',limit=2).interpolate_na(dim='hour',limit=1)

#%% Plots
plt.close('all')
fig=plt.figure(figsize=(16,8))

#plot T,r variables
ctr=1
for v in variables:
    if v!='pblh':
        ax=fig.add_subplot(2,2,ctr)
        cf=plt.contourf(hour_avg,Data_avg.height,Data_avg[v].T,ticks[v],cmap=colormaps[v],extend='both')
        plt.contour(hour_avg,Data_avg.height,Data_avg[v].T,ticks[v],colors='k',linewidths=0.5,linestyles='solid',alpha=0.5)
        if ctr==1:
            pblh=Data_avg['pblh'].isel(height=0)*1000
            plt.plot(hour_avg[hour_avg>=12],pblh[hour_avg>=12],'.w',markersize=20,markeredgecolor='k')
        if ctr==1 or ctr==3:
            plt.ylabel(r'$z$ [m a.g.l.]')
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0,6,12,18,24])
        ax.set_xticklabels(['0000','0600','1200','1800','2400'])
        if ctr>=3:
            plt.xlabel('Hour (UTC)')
        else:
            ax.set_xticklabels([])
        
        plt.xlim([0,24])
        plt.ylim([0,max_height])
        plt.grid()
        ctr+=1
        plt.colorbar(cf,label=labels[v],ticks=ticks[v][::4])
plt.tight_layout()

plt.figure()
v='dtheta_v_dz'
cf=plt.contourf(hour_avg,Data_avg.height,Data_avg[v].T,ticks[v],cmap=colormaps[v],extend='both')
plt.contour(hour_avg,Data_avg.height,Data_avg[v].T,ticks[v],colors='k',linewidths=0.5,linestyles='solid',alpha=0.5)
plt.xlim([0,24])
plt.ylim([0,200])
plt.plot([0,24],[H-D/2,H-D/2],'--k')
plt.plot([0,24],[H+D/2,H+D/2],'--k')
plt.xticks([0,6,12,18,24])
ax.set_xticklabels(['0000','0600','1200','1800','2400'])
plt.grid()
