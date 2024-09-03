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
from datetime import datetime
from scipy import signal
from scipy import stats

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
source_config=os.path.join(cd,'config.yaml')

#dataset
# source_met='awaken/sa1.met.z01.c0/*nc'
source_lid=os.path.join(cd,'data/sa1.lidar.z03.c1.20230724.20230801.nc')
source_ast=os.path.join(cd,'data/sb.assist.z01.c0.20230724.20230801.nc')
source_met=os.path.join(cd,'data/sa2.met.z01.c0.20230724.20230801.nc')
source_snc=os.path.join(cd,'data/sa2.sonic.z01.c0.20230724.20230801.nc')

#user-defined
variables_lid=['WS','WD','TKE']
variables_ast=['temperature','rh']
variables_met=['average_wind_speed','wind_direction','temperature','relative_humidity']
variables_snc=['TKE']
DT=1800#[s] common sampling time

#qc
max_TKE=10#[m^2/s^2] maximum TKE
min_lwp=10#[g/m^2] for liquid water paths lower than this, remove clouds
max_gamma=5#maximum weight of the prior
max_rmsr=5#maximum rms of the retrieval

#graphics

variables={'WS':'average_wind_speed','WD':'wind_direction','TKE':'TKE','temperature':'temperature','rh':'relative_humidity'}


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
AST=AST.where(AST['height']<AST['cbh']).where(AST['rmsr']<max_rmsr).where(AST['gamma']<max_gamma)

for v in variables_met:
    MET[v]=MET[v].where(MET['qc_'+v]==0)

SNC=SNC.where(SNC['QC flag']==0)

#interpolation
for v in variables_lid:
    LID[v]=LID[v].interpolate_na(dim='time',method='linear')
for v in variables_ast:
    AST[v]=AST[v].interpolate_na(dim='time',method='linear')
for v in variables_met:
    MET[v]=MET[v].interpolate_na(dim='time',method='linear')
for v in variables_snc:
    SNC[v]=SNC[v].interpolate_na(dim='time',method='linear')

#psd of remote sensing
for v in variables_lid+variables_ast: 
    psd_T=[]
    period=[]
    for h in height:
        print(v+': z = '+str(h)+' m') 
        
        if v in variables_lid:
            f_sel=LID[v].sel(height=h)
        elif v in variables_ast:
            f_sel=AST[v].sel(height=h)
        
        tnum= [utl.dt64_to_num(t) for t in f_sel.time.values]
        tnum_res=np.arange(np.min(tnum),np.max(tnum)+1,DT)
        f=np.interp(tnum_res,tnum,f_sel.values)
        
        f_psd,psd=signal.periodogram(f, fs=1/DT,  scaling='density')
    
        period=utl.vstack(period,(1/f_psd))
        psd_T=utl.vstack(psd_T,(psd/np.var(f)*f_psd**2))
    
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
    
    f_psd,psd=signal.periodogram(f, fs=1/DT,  scaling='density')

    PSD_sfc[v]=(psd/np.var(f)*f_psd**2)
    T_sfc[v]=(1/f_psd)
    
#%% Plots
plt.close('all')
ctr=1
fig=plt.figure(figsize=(12,10))

for v in variables:
    ax=plt.subplot(len(variables),1,ctr)
    ctr2=0
    for i_h in range(len(height)):
        plt.loglog(T[v][i_h,:]/3600,PSD[v][i_h,:],label=r'$z='+str(int(height[i_h]))+'$ m',color=colors[ctr2],alpha=0.75)
        ctr2+=1
    plt.xticks([6,12,24,48],labels=['6','12','24','48'])
    plt.xlim([2,100])
    plt.ylim([10**-8,10**-4])
   
    if ctr==len(variables):
        plt.xlabel('Period [hours]')
    else:
        ax.set_xticklabels([])
    plt.ylabel('PSD [hours$^{-1}$]')
    plt.loglog(T_sfc[variables[v]]/3600,PSD_sfc[variables[v]],label='Surface',color='k',alpha=1)
    plt.title(v)
    plt.grid()
    
    ctr+=1

plt.tight_layout()
plt.legend()