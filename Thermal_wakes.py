# -*- coding: utf-8 -*-
"""
Calculated psd of inflow variables
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('./utils')
import utils as utl
import numpy as np
import pandas as pd
import matplotlib.patheffects as path_effects
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib as mpl
import glob
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

mpl.rcParams.update({
"savefig.format": "pdf",
"pdf.fonttype": 42,
"ps.fonttype": 42,
"font.family": "serif",
"font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
"mathtext.fontset": "custom",
"mathtext.rm": "serif",
"mathtext.it": "serif:italic",
"mathtext.bf": "serif:bold",
"axes.labelsize": 14,
"axes.titlesize": 14,
"xtick.labelsize": 12,
"ytick.labelsize": 12,
"legend.fontsize": 12,
"lines.linewidth": 1,
"lines.markersize": 4,
})

#%% Inputs

sites=['Middle','North']#selected sites
sites_awaken={'Middle':'C1a','North':'G'}#AWAKEN site name

#TROPoe data paths
sources_trp={'South': 'C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/awaken_processing/data/awaken/sb.assist.tropoe.z01.c0/*nc',
             'Middle':'C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/awaken_processing/data/awaken/sc1.assist.tropoe.z01.c0/*nc',
             'North': 'C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/ASSIST_analysis/awaken_processing/data/awaken/sg.assist.tropoe.z01.c0/*nc'}

#sonic data paths
sources_snc={'A2':os.path.join(cd,'data/sa2.sonic.z01.c0.20230101.20240101.nc'),
             'A5':os.path.join(cd,'data/sa5.sonic.z01.c0.20230101.20240101.nc')}

source_inf=os.path.join(cd,'data/glob.lidar.eventlog.avg.c2.20230101.000500.csv')#source event log

source_layout=os.path.join(cd,'data','20250225_AWAKEN_layout.nc')#source layout

#stats
max_height=500#[m] maximum height
min_lwp=5#[g/cm^2] minimum liquid water path to flag clouds
max_gamma=1#maximum convergence factor
max_rmsa=5#maximum RMS of retrieval
bin_wd=np.arange(0,361,20) #bins in wind direction
perc_lim=[5,95]#[%] percentile limits
p_value=0.05#p-value for c.i.
bin_ws=np.array([0,4,15,20])

#wakes
N_wakes=24#number of directions
max_dist=10000#[m] maximum distance
H=89#[m] hub height
D_ref=127#[m] diameter

#unwaked sector for sonics
wd_range_snc={'A2':[0,270],
              'A5':[270,360]}
#stability classes
stab_classes={'S':[0,np.inf],
            'U':[-np.inf,0]}

stab_class_uni=['S','U']

#graphics
colors={'North':(0,0.6,0,1),'Middle':(0.8,0.4,0,1)}
cmap_wf = LinearSegmentedColormap.from_list("my_cmap", [(0.8,0.4,0,1),(0,0.6,0,1)])

#%% Functions
def nanmean_dataset(ds1, ds2):
    '''
    Variable-wise mean of datasets
    '''
    avg_dict = {}
    ds1_synch,ds2_synch=xr.align(ds1,ds2,join='outer')
    for var in ds1.data_vars:
        avg_dict[var] = xr.DataArray(np.nanmean([ds1_synch[var], ds2_synch[var]], axis=0),
                                    dims=ds1_synch[var].dims, coords=ds1_synch[var].coords)
    return xr.Dataset(avg_dict)

#%% Initalization

#read and QC all TROPoe data
save_path=os.path.join(cd,'data','TROPoe_all.nc')
if not os.path.isfile(save_path):
    Data_trp=xr.Dataset()
    for site in sites:
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
        
        Data_trp[f'T_{site}']=Data.temperature
        Data_trp[f'sigma_T_{site}']=Data.sigma_temperature
        Data.close()
    Data_trp.to_netcdf(save_path)
    Data_trp.close()

Data_trp=xr.open_dataset(save_path)

#load inflow
inflow_df=pd.read_csv(source_inf).set_index('UTC Time')
inflow_df.index= pd.to_datetime(inflow_df.index)
Inflow=xr.Dataset.from_dataframe(inflow_df).rename({'UTC Time':'time'})

#load sonic
SNC={}
SNC_unw={}
for s in sources_snc:
    SNC[s]=xr.open_dataset(os.path.join(cd,sources_snc[s]))
    SNC[s]=SNC[s].where(SNC[s]['QC flag']==0).interp(time=Data_trp.time)
    
    #rejected sectors affected by container
    if wd_range_snc[s][1]>wd_range_snc[s][0]:
       SNC_unw[s]=SNC[s].where(SNC[s]['wind direction']>=wd_range_snc[s][0]).where(SNC[s]['wind direction']<wd_range_snc[s][1])
    else:
       SNC_unw[s]=SNC[s].where((SNC[s]['wind direction']<wd_range_snc[s][1]) | (SNC[s]['wind direction']>=wd_range_snc[s][0]))

#load layout 
Turbines=xr.open_dataset(source_layout,group='turbines')
Sites=xr.open_dataset(source_layout,group='ground_sites')

#%% Main

#wake count
wd_wakes=np.linspace(0,360,N_wakes+1)[:-1]
xt=Turbines.x_utm.values
yt=Turbines.y_utm.values
D=Turbines.Diameter.values
D[np.isnan(D)]=D_ref
waking={}
for site in sites:
    waking[site]=np.zeros((len(wd_wakes),len(xt)))==1
    x0=np.float64(Sites.x_utm.sel(site=sites_awaken[site]))
    y0=np.float64(Sites.y_utm.sel(site=sites_awaken[site]))
    L=((xt-x0)**2+(yt-y0)**2)**0.5
    sel=np.where(L<max_dist)[0]
    wd_align=(90-np.degrees(np.arctan2(yt[sel]-y0,xt[sel]-x0)))%360
    waked_sector=1.3*np.degrees(np.arctan(2.5*D[sel]/L[sel]+0.15))+10#[IEC 61400-12-5:2022] waked sector
    
    for i_wd in range(len(wd_wakes)):
        wd_diff=(wd_align - wd_wakes[i_wd] + 180) % 360 - 180
        waking[site][i_wd,sel[np.abs(wd_diff)<waked_sector/2]]=True

wake_factor=np.sum(waking['North'],axis=1)-np.sum(waking['Middle'],axis=1)
    
#temperature difference
diff=Data_trp['T_North']-Data_trp['T_Middle']

#interpolate wind conditions
ws=Inflow['Hub-height wind speed [m/s]'].interp(time=Data_trp.time)

cos=np.cos(np.radians(Inflow['Hub-height wind direction [degrees]'].interp(time=Data_trp.time)))
sin=np.sin(np.radians(Inflow['Hub-height wind direction [degrees]'].interp(time=Data_trp.time)))
wd=np.degrees(np.arctan2(sin,cos))%360

#stability
SNC_cmb = nanmean_dataset(SNC_unw[list(sources_snc.keys())[0]], SNC_unw[list(sources_snc.keys())[1]]).sortby('time')

#stab classes
SNC_cmb['Stability class']=xr.DataArray(data=['null']*len(SNC_cmb.time),coords={'time':SNC_cmb.time})

for s in stab_classes.keys():
    sel=(SNC_cmb['Obukhov\'s length']>=stab_classes[s][0])*(SNC_cmb['Obukhov\'s length']<stab_classes[s][1])
    if s=='N1' or s=='N2':
        s='N'
    SNC_cmb['Stability class']=SNC_cmb['Stability class'].where(~sel,other=s)
    
#ensamble stats
ws_avg=(bin_ws[:-1]+bin_ws[1:])/2
wd_avg=(bin_wd[:-1]+bin_wd[1:])/2
diff_stats=xr.Dataset()
for stab_class in stab_class_uni:
    f_avg_all=np.zeros((len(ws_avg),len(wd_avg),len(diff.height)))
    f_low_all=np.zeros((len(ws_avg),len(wd_avg),len(diff.height)))
    f_top_all=np.zeros((len(ws_avg),len(wd_avg),len(diff.height)))
    
    sel_stab=SNC_cmb['Stability class']==stab_class
    
    for i_h in range(len(diff.height)):
        f_sel=diff.where(sel_stab).isel(height=i_h).values
    
        real=~np.isnan(f_sel+wd.values)
        f_avg= stats.binned_statistic_2d(ws.values[real], wd.values[real], f_sel[real],statistic=lambda x: utl.filt_stat(x,   np.nanmean,perc_lim=perc_lim),                          bins=[bin_ws,bin_wd])[0]
        f_low= stats.binned_statistic_2d(ws.values[real], wd.values[real], f_sel[real],statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=p_value/2*100),    bins=[bin_ws,bin_wd])[0]
        f_top= stats.binned_statistic_2d(ws.values[real], wd.values[real], f_sel[real],statistic=lambda x: utl.filt_BS_stat(x,np.nanmean,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=[bin_ws,bin_wd])[0]
    
        f_avg_all[:,:,i_h]=f_avg
        f_low_all[:,:,i_h]=f_low
        f_top_all[:,:,i_h]=f_top
        
        print(f'Stability class {stab_class}: Level {i_h} done')
   
    diff_stats[f'diff_avg_{stab_class}']=xr.DataArray(data=f_avg_all,coords={'ws':ws_avg,'wd':wd_avg,'height':Data_trp.height.values})
    diff_stats[f'diff_low_{stab_class}']=xr.DataArray(data=f_low_all,coords={'ws':ws_avg,'wd':wd_avg,'height':Data_trp.height.values})
    diff_stats[f'diff_top_{stab_class}']=xr.DataArray(data=f_top_all,coords={'ws':ws_avg,'wd':wd_avg,'height':Data_trp.height.values})
    diff_stats[f'diff_qc_{stab_class}']=diff_stats[f'diff_avg_{stab_class}'].where(diff_stats[f'diff_top_{stab_class}']-diff_stats[f'diff_low_{stab_class}']<=np.abs(diff_stats[f'diff_avg_{stab_class}']),0)

#assess wake predictor
diff_int=diff_stats['diff_qc_S'].isel(height=0,ws=1).interp(wd=wd_wakes).values
print(f'Correlation coefficient at the ground for max_dist = {max_dist} m in Stable: {np.corrcoef(diff_int[~np.isnan(diff_int)],wake_factor[~np.isnan(diff_int)])[0][1]:03f}')

diff_int=diff_stats['diff_qc_U'].isel(height=0,ws=1).interp(wd=wd_wakes).values
print(f'Correlation coefficient at the ground for max_dist = {max_dist} m in Unstable: {np.corrcoef(diff_int[~np.isnan(diff_int)],wake_factor[~np.isnan(diff_int)])[0][1]:03f}')


#%% Plots
plt.close('all')
theta=np.radians(np.append(wd_avg,wd_avg[0]+360))
theta2=np.radians(np.arange(360))
for i_ws in range(len(ws_avg)):
    fig, axs = plt.subplots(1, len(stab_class_uni), figsize=(18, 8), constrained_layout=True,subplot_kw={'projection': 'polar'})
    i_s=0
    for stab_class in stab_class_uni:
        plt.sca(axs[i_s])
        ax=axs[i_s]
        f_plot_qc=np.vstack([diff_stats[f'diff_qc_{stab_class}'].values[i_ws,:,:],
                             diff_stats[f'diff_qc_{stab_class}'].values[i_ws,0,:]]).T
        cf=plt.contourf(theta,diff.height*1000+500,f_plot_qc,np.arange(-1,1.01,0.02),extend='both',cmap='seismic')
        plt.contour(theta,diff.height*1000+500,f_plot_qc,np.arange(-1,1.01,0.02),colors='k',alpha=0.5,linewidths=0.1)
        plt.plot(theta2,theta2*0+H+D_ref/2+500,'k')
        
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rlim([0,500+max_height])
        ax.set_yticklabels([])
        cax=plt.colorbar(cf,label=r'Temperature difference (G-C1a) [$^\circ$C]')
        cax.set_ticks(np.arange(-1,1.1,0.25))
        plt.scatter(np.radians(wd_wakes),300+np.zeros(len(wd_wakes)),s=350,c=(wake_factor+25)/41*255,cmap=cmap_wf,edgecolor='k',vmin=0,vmax=255)
        for i_wd in range(len(wd_wakes)):
            plt.text(np.radians(wd_wakes[i_wd]), 300, wake_factor[i_wd],
                     ha='center', va='center',color='w', fontsize=10, weight='bold',
                     path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
        ax.grid(False)
        ax.set_xticks([0,np.pi/2,np.pi,np.pi/2*3], labels=['N','E','S','W'])
        i_s+=1

for i_wd in range(len(wd_wakes)):

    plt.figure()
    plt.scatter(xt,yt,s=21,c='w',edgecolor='k')
    for site in sites:
        plt.scatter(xt[waking[site][i_wd,:]],yt[waking[site][i_wd,:]],s=20,c=colors[site],alpha=0.5)
                                             
        plt.scatter(Sites.x_utm.sel(site=sites_awaken[site]),
                 Sites.y_utm.sel(site=sites_awaken[site]),s=100,c=colors[site],marker='*',edgecolor='k')  
    plt.xlim([611346., 647779.])                     
    plt.ylim([4009878., 4036982.])
    plt.gca().set_aspect('equal')
    plt.savefig(os.path.join(cd,'figures',f'wakes_{int(wd_wakes[i_wd]):03}.png'))
    plt.close()
    