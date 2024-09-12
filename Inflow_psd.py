# -*- coding: utf-8 -*-
"""
Calculated psd of inflow variables 
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
import yaml
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
from scipy import signal

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source_config=os.path.join(cd,'config.yaml')

#dataset
source_lid=os.path.join(cd,'data/sa1.lidar.z03.c1.20230101.20240101.5_levels.nc')
source_ast=os.path.join(cd,'data/sb.assist.z01.c0.20230101.20240101.5_levels.nc')
source_met=os.path.join(cd,'data/sa2.met.z01.c0.20230101.20240101.nc')
source_snc=os.path.join(cd,'data/sa2.sonic.z01.c0.20230101.20240101.nc')

#user-defined
variables_lid=['WS','WD','TKE']
variables_ast=['temperature','waterVapor']
variables_met=['average_wind_speed','wind_direction','temperature','waterVapor']
variables_snc=['TKE']
DT=1800#[s] common sampling time
N_windows=5#windows of Welch
option='welch'#type of fft
max_nans_gap=np.timedelta64(7*24, 'h')#maximum consecutive nans

#physics
mm_v=18.015#[g/mol] molecular mass of water
mm_a=28.96#[g/mol] molecular mass of air

#qc
max_TKE=10#[m^2/s^2] maximum TKE
min_lwp=10#[g/m^2] for liquid water paths lower than this, remove clouds
max_gamma=5#maximum weight of the prior
max_rmsr=5#maximum rms of the retrieval

#graphics
variables={'WS':'average_wind_speed','WD':'wind_direction','TKE':'TKE','temperature':'temperature','waterVapor':'waterVapor'}
T_plot=np.arange(1000*3600)+0.0
height=[]

#%% Function
def spectrum(x,DT,option,N_windows):
    if option=='fft':
        f,p0=signal.periodogram(x,fs=1/DT,scaling='density')
    elif option=='welch':
        f, p0= signal.welch(x, fs=1/DT, nperseg=int(len(x)/N_windows))
        
    p=p0*f**2
    p_norm=-p/np.trapz(p[1:],1/f[1:])
    return f,p_norm

def vapor_pressure(T):
    e=T*0
    A=7.5
    B=237.3
    e[T>=0]=6.11*10**((A*T[T>=0])/(T[T>=0]+B))*100
    
    A=9.5
    B=265.5
    e[T<0]=6.11*10**((A*T[T<0])/(T[T<0]+B))*100

    return e

#%% Initialization

#config
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
import utils as utl

#load data
LID=xr.open_dataset(source_lid)
AST=xr.open_dataset(source_ast)
MET=xr.open_dataset(source_met)
SNC=xr.open_dataset(source_snc)

#zeroing
PSD={}
T={}
PSD_sfc={}
T_sfc={}

#graphics
if height==[]:
    height=LID.height.values
colormap = plt.cm.viridis
colors = [colormap(i) for i in np.linspace(0,1,len(height))]

#%% Main 

#qc
TKE_qc=LID['TKE'].where(LID['TKE']>0).where(LID['TKE']<max_TKE)
original_nans=np.isnan(LID['TKE'])
TKE_int=TKE_qc.chunk({"time": -1}).interpolate_na(dim='time',method='linear')
TKE_int=TKE_int.where(original_nans==False)
LID['TKE']=TKE_int

AST['height']=AST.height*1000
AST['cbh']=AST.cbh*1000
AST['cbh'][AST['lwp']<min_lwp]=10000

for v in variables_ast: 
    AST[v]=AST[v].where(AST['height']<AST['cbh']).where(AST['rmsr']<max_rmsr).where(AST['gamma']<max_gamma)

for v in MET.data_vars:
    try:
        MET[v]=MET[v].where(MET['qc_'+v]==0)
    except:
        pass

SNC=SNC.where(SNC['QC flag']==0).sortby('time')


#mixing ratio
MET['saturated_vapor_pressure']=vapor_pressure(MET['temperature'])
MET['waterVapor']=mm_v/mm_a*MET['relative_humidity']/100*MET['saturated_vapor_pressure']/(MET['pressure']*1000-MET['relative_humidity']/100*MET['saturated_vapor_pressure'])*1000

#interpolation
max_nans_edges=int(np.int(max_nans_gap/np.median(np.diff(LID.time))))
for v in variables_lid:
    LID[v]=LID[v].interpolate_na(dim='time',method='linear',max_gap=max_nans_gap).ffill(dim="time",limit=max_nans_edges).bfill(dim="time",limit=max_nans_edges)
max_nans_edges=int(np.int(max_nans_gap/np.median(np.diff(AST.time))))
for v in variables_ast:
    AST[v]=AST[v].interpolate_na(dim='time',method='linear',max_gap=max_nans_gap).ffill(dim="time",limit=max_nans_edges).bfill(dim="time",limit=max_nans_edges)
max_nans_edges=int(np.int(max_nans_gap/np.median(np.diff(MET.time))))
for v in variables_met:
    MET[v]=MET[v].interpolate_na(dim='time',method='linear',max_gap=max_nans_gap).ffill(dim="time",limit=max_nans_edges).bfill(dim="time",limit=max_nans_edges)
max_nans_edges=int(np.int(max_nans_gap/np.median(np.diff(SNC.time))))
for v in variables_snc:
    SNC[v]=SNC[v].interpolate_na(dim='time',method='linear',max_gap=max_nans_gap).ffill(dim="time",limit=max_nans_edges).bfill(dim="time",limit=max_nans_edges)
    
#wind direction unwrapping
LID['WD'].values=np.unwrap(LID.WD,period=360,axis=0)
MET['wind_direction'].values=np.unwrap(MET.wind_direction,period=360,axis=0)

#psd of remote sensing
for v in variables_lid+variables_ast: 
    psd_T=[]
    period=[]
    for h in height:
        print(v+': z = '+str(h)+' m') 
        
        if v in variables_lid:
            f_sel=LID[v].sel(height=h,method='nearest')
        elif v in variables_ast:
            f_sel=AST[v].sel(height=h,method='nearest')
        
        tnum= [utl.dt64_to_num(t) for t in f_sel.time.values]
        tnum_res=np.arange(np.min(tnum),np.max(tnum)+1,DT)
        f=np.interp(tnum_res,tnum,f_sel.values)
        if np.sum(np.isnan(f))==0:
            f_det=signal.detrend(f)
        else:
            f_det=f.copy()
        f_psd,psd=spectrum(f_det, DT,option,N_windows)
    
        period=utl.vstack(period,(1/f_psd))
        psd_T=utl.vstack(psd_T,psd)
    
    PSD[v]=psd_T
    T[v]=period
    
#psd of surface sensors
for v in variables_met+variables_snc: 
    if v in variables_met:
        f_sel=MET[v]
    elif v in variables_snc:
        f_sel=SNC[v]
    
    tnum= [utl.dt64_to_num(t) for t in f_sel.time.values]
    tnum_res=np.arange(np.min(tnum),np.max(tnum)+1,DT)
    f=np.interp(tnum_res,tnum,f_sel.values)
    
    if np.sum(np.isnan(f))==0:
        f_det=signal.detrend(f)
    else:
        f_det=f.copy()
    f_psd,psd=spectrum(f_det, DT,option,N_windows)

    PSD_sfc[v]=psd
    T_sfc[v]=1/f_psd
    
#%% Plots

#colormap
plt.close('all')
for v in variables:
    plt.figure()
    plt.pcolor(T[v][0,1:]/3600,height,np.log10(PSD[v][:,1:]),cmap='hot',vmin=-8,vmax=-2)
    ax=plt.gca()
    ax.set_xscale('log')
    plt.xlim([2,1000])
    plt.ylim([0,3000])
    plt.colorbar(label=v)
    plt.xlabel('Period [hour]')
    plt.ylabel(r'$z$ [m.a.g.l.]')
    
#linear plot

    
if len(height)<10:
    ctr=1
    fig=plt.figure(figsize=(18,10))
    
    for v in variables:

        for i_h in range(len(height)):
            ax=plt.subplot(1,len(variables),ctr)
            plt.loglog(T[v][i_h,:]/3600,PSD[v][i_h,:]*10**(i_h*3),color='k',alpha=1,linewidth=1)
            if ctr==1:
                plt.text(30,PSD[v][i_h,-1]*10**(i_h*3),str(int(height[i_h]))+' m AGL',bbox={'alpha':0.1,'facecolor':'w'})
            if i_h==0:
                plt.loglog(T_sfc[variables[v]]/3600,PSD_sfc[variables[v]]*10**(-3),label='Surface',color='k',alpha=1,linewidth=1)
                if ctr==1:
                    plt.text(30,PSD_sfc[variables[v]][-1]*10**(-3),'Surface',bbox={'alpha':0.1,'facecolor':'w'})
            plt.xticks([6,12,24,24*4])
            plt.xlim([2,1000])
            # plt.ylim([10*-8,10])
            plt.yticks([])
            ax.set_xticklabels(['6 h','12 h','1 day','4 days'],rotation=30)
            plt.grid()
        ctr+=1
    
    plt.tight_layout()