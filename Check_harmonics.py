# -*- coding: utf-8 -*-
"""
Check high harmonics
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
from scipy import signal
from scipy import stats

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18


#%% Inputs
source=os.path.join(cd,'data/sa2.met.z01.c0.20230101.20240101.nc')

bin_tnum=np.arange(0,24*3600+1,600)
repeat=20
#%% Initialization
Data=xr.open_dataset(source)
tnum=np.float64(Data.time)/10**9

DT=np.median(np.diff(tnum))

#%% Main

f=Data['temperature'].values#.interpolate_na(dim='time',method='linear').values
f_da=Data['temperature'].rolling(time=int(24*3600/DT),center=True).mean().interpolate_na(dim='time',method='linear').values

f_real=f[~np.isnan(f-f_da)]
f_da_real=f_da[~np.isnan(f-f_da)]
tnum_real=tnum[~np.isnan(f-f_da)]
time_real=Data.time[~np.isnan(f-f_da)]

real2=np.where(np.diff(tnum_real)>DT*1.1)[0]
real1=np.where(np.diff(tnum_real)>DT*1.1)[0]+1
if ~np.isnan(f[0]):
    real1=np.append(0,real1)
if ~np.isnan(f[-1]):
    real2=np.append(real2,len(f))
longest=np.argmax(real2-real1)

f_sel=f_real[real1[longest]:real2[longest]]
tnum_sel=tnum_real[real1[longest]:real2[longest]]
time_sel=time_real[real1[longest]:real2[longest]]
f_da_sel=f_da_real[real1[longest]:real2[longest]]


f_psd,psd=signal.periodogram(f_sel, fs=1/DT,  scaling='density')

tnum_day=np.mod(tnum_sel,utl.floor(tnum_sel,24*3600))
f_avg=stats.binned_statistic(tnum_day, f_sel-f_da_sel,statistic='mean',bins=bin_tnum)[0]
tnum_avg=utl.mid(bin_tnum)

f_avg_rep=np.tile(f_avg,(1,repeat)).squeeze()

DT_avg=np.median(np.diff(tnum_avg))
f_psd_avg,psd_avg=signal.periodogram(f_avg_rep, fs=1/DT_avg,  scaling='density')

#%% Plots
plt.figure()
plt.loglog(f_psd,psd,'k',label='Raw signal')
plt.loglog(f_psd_avg,psd_avg,'r',label='Idealized cycle')
plt.xlabel('Frequency')
plt.ylabel('Spectrum')
plt.grid()
plt.legend()

plt.figure()
plt.plot(time_sel,f_sel,'k',label='Raw signal')
plt.plot(time_sel,f_da_sel,'r',label='24-h average')
plt.plot(time_sel,f_sel-f_da_sel,'b',label='Difference')
plt.xlabel('Time (UTC)')
plt.ylabel('T [degrees C]')
plt.legend()
plt.grid()

plt.figure()
plt.plot(f_avg_rep,'k')

plt.grid()
