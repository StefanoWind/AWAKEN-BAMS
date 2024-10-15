# -*- coding: utf-8 -*-
"""
Daily statisitcs from ASSISTs+TROPoe
"""
import os
cd=os.path.dirname(__file__)
import sys
import numpy as np
import yaml
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
from scipy import stats

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source_config=os.path.join(cd,'config.yaml')

#dataset
source=os.path.join(cd,'data/sb.assist.z01.c0.20230101.20240101.41_levels.nc')

#site
H=89#[m] hub height
e=0.622#ratio of molecular mass (water to air)
g=9.81#[m/s^2] gravity constant
cp=1005#[J/KgK] air specific heat

#qc
min_lwp=10#[g/m^2] for liquid water paths lower than this, remove clouds
max_gamma=5#maximum weight of the prior
max_rmsr=5#maximum rms of the retrieval

#stats
bin_month=np.arange(0,12.1,3)
bin_hour=np.arange(25)
perc_lim=[5,95]#[%] outlier rejection
p_value=0.05#p-value for confidence interval
max_err={'temperature':4,'waterVapor':2,'dtheta_v_dz':2}

#user
variables=['temperature','waterVapor','dtheta_v_dz','perc_stable']

#graphics
labels={'temperature':r'$T$ [$^\circ$C]','waterVapor':r'$r$ [g Kg$^{-1}$]',
        'dtheta_v_dz':r'$\frac{\partial \theta_v}{\partial z}$ [$^\circ$C m$^{-1}$]','perc_stable':r'Occurrence of $\frac{\partial \theta_v}{\partial z}>0$ [%]'}
ticks={'temperature':np.arange(15,31),'waterVapor':np.arange(5,15,0.5),'dtheta_v_dz':np.arange(-0.05,0.051,0.0025),'perc_stable':np.arange(0,101,10)}
colormaps={'temperature':'coolwarm','waterVapor':'Blues','dtheta_v_dz':'seismic','perc_stable':'viridis'}

#%% Initialization
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl

#load data
AST=xr.open_dataset(source)

#qc
AST['height']=AST.height*1000
AST['cbh']=AST.cbh*1000
AST['cbh'][AST['lwp']<min_lwp]=10000

#virtual temperature
r=AST['waterVapor']/1000
AST['theta_v']=AST['theta']*(e+r)/(e*(1+r))-273.15
AST['dtheta_v_dz']=AST['theta_v'].differentiate('height')

for v in variables[:-1]: 
    AST[v]=AST[v].where(AST['height']<AST['cbh']).where(AST['rmsr']<max_rmsr).where(AST['gamma']<max_gamma)
    
#data extraction
height=AST.height.values
tnum=np.float64(AST.time)/10**9
hour=np.mod(tnum,3600*24)/3600#time of the day
hour_avg=utl.mid(bin_hour)

#zeeroing
AST_avg={}
AST_int=xr.Dataset()

#%% Main

#daily average
AST_avg=xr.Dataset()
for v in variables[:-1]:
    f_avg_all=[]
    for h in height:
        print(v+': '+str(h))
        f=AST[v].sel(height=h).values
        f[f==0]=np.nan
        real=~np.isnan(f)
        
        if np.sum(real)>0:
            f_avg= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_mean(x,perc_lim=perc_lim),                             bins=bin_hour)[0]
            f_low= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_BS_mean(x,perc_lim=perc_lim,p_value=p_value/2*100),    bins=bin_hour)[0]
            f_top= stats.binned_statistic(hour[real], f[real],statistic=lambda x:utl.filt_BS_mean(x,perc_lim=perc_lim,p_value=(1-p_value/2)*100),bins=bin_hour)[0]

            f_avg[np.abs(f_top-f_low)>max_err[v]]=np.nan
            f_avg[np.isnan(np.abs(f_top-f_low))]=np.nan
            
            f_avg_all=utl.vstack(f_avg_all,f_avg)

        else:
            f_avg_all=utl.vstack(f_avg_all,hour_avg*np.nan)
       
    AST_avg[v]=xr.DataArray(data=f_avg_all.T,coords={'hour':hour_avg,'height':AST.height.values}).interpolate_na(dim='height',limit=2).interpolate_na(dim='hour',limit=1)

perc_stable=[]
AST['stable']=AST['dtheta_v_dz']>0
AST['unstable']=AST['dtheta_v_dz']<0
for h in height:
    stab_sel=AST['stable'].sel(height=h).values
    unst_sel=AST['unstable'].sel(height=h).values
    real=~np.isnan(stab_sel+unst_sel)
    if np.sum(real)>0:
        N_stab=stats.binned_statistic(hour[real], stab_sel[real],statistic='sum',bins=bin_hour)[0]
        N_unst=stats.binned_statistic(hour[real], unst_sel[real],statistic='sum',bins=bin_hour)[0]
        perc_stable=utl.vstack(perc_stable,N_stab/(N_stab+N_unst)*100)
AST_avg['perc_stable']=xr.DataArray(data=perc_stable.T,coords={'hour':hour_avg,'height':AST.height.values}).interpolate_na(dim='height',limit=2).interpolate_na(dim='hour',limit=1)
    
#%% Plots
plt.close('all')
fig=plt.figure(figsize=(16,8))

#plot T,r variables
ctr=1
for v in variables[:2]:
    ax=fig.add_subplot(2,1,ctr)
    cf=plt.contourf(hour_avg,AST_avg.height,AST_avg[v].T,ticks[v],cmap=colormaps[v],extend='both')
    plt.contour(hour_avg,AST_avg.height,AST_avg[v].T,ticks[v],colors='k',linewidths=0.5,linestyles='solid')

    plt.ylabel(r'$z$ [m AGL]')
        
    ax.set_xticks([0,6,12,18,24])
    if ctr==2:
        plt.xlabel('Hour (UTC)')
    else:
        ax.set_xticklabels([])
        
    plt.grid()
    ctr+=1
        
    plt.colorbar(cf,label=labels[v])
plt.tight_layout()

fig=plt.figure(figsize=(16,8))

#plot theta, perc_stable
ctr=1
for v in variables[2:]:
    ax=fig.add_subplot(2,1,ctr)
    cf=plt.contourf(hour_avg,AST_avg.height,AST_avg[v].T,ticks[v],cmap=colormaps[v],extend='both')
    plt.contour(hour_avg,AST_avg.height,AST_avg[v].T,ticks[v],colors='k',linewidths=0.5,linestyles='solid')

    plt.ylabel(r'$z$ [m AGL]')
        
    ax.set_xticks([0,6,12,18,24])
    if ctr==2:
        plt.xlabel('Hour (UTC)')
    else:
        ax.set_xticklabels([])
        
    plt.grid()
    ctr+=1
        
    plt.colorbar(cf,label=labels[v])
plt.tight_layout()


#profiles
fig=plt.figure(figsize=(16,6))
ax=fig.add_subplot(111)
scale=0.075
for h in hour_avg:
    ax.fill_between([h*scale-g/cp,h*scale+g/cp],[500,500],0,color='k',alpha=0.25)
    plt.plot([h*scale,h*scale],[0,500],'--k')
    plt.plot(AST_avg['dtheta_v_dz'].sel(hour=h)+h*scale,AST_avg.height,'-',color='k',markersize=12)
    plt.plot(AST_avg['dtheta_v_dz'].where(AST_avg['dtheta_v_dz']>0).sel(hour=h)+h*scale,AST_avg.height,'.',color='b',markersize=12)
    plt.plot(AST_avg['dtheta_v_dz'].where(AST_avg['dtheta_v_dz']<0).sel(hour=h)+h*scale,AST_avg.height,'.',color='r',markersize=12)
    
plt.ylim([0,500])
plt.grid()
plt.xticks(np.arange(24*scale))
plt.ylabel(r'$z$ [m AGL]')
plt.xticks(np.arange(25)*scale,labels=np.arange(25))
plt.xlabel('Hour (UTC)')


