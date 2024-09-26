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
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib
from scipy import stats
import matplotlib.gridspec as gridspec

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
source_all=os.path.join(cd,'data','inflow_all.nc')
source_avg=os.path.join(cd,'data','inflow_avg_{v}.nc')

months={'DJF':[0,2],'MAM':[3,5],'JJA':[6,8],'SON':[9,11]}


#%% Initialiation
ALL=xr.open_dataset(source_all)
ALL_avg={}
for m in months:
    ALL_avg[m]=xr.open_dataset(source_avg.format(v=m))
    
hour=np.array([int(str(t)[11:13])+int(str(t)[14:16])/60 for t in ALL.time.values])    
# ALL['hour']=xr.DataArray(data=hour,coords={'time':ALL.time.values})

ALL_ref={}

#%% Main
for m in months:
 
    ALL_ref[m]=ALL_avg[m].interp({'hour':hour}).assign_coords(hour=ALL.time.values).rename({'hour':'time'})
    ALL_ref[m]['month']=xr.DataArray(data=ALL['month'].values,coords={'time':ALL.time.values})
    ALL_ref[m]=ALL_ref[m].where(ALL_ref[m]['month']>=months[m][0]).where(ALL_ref[m]['month']<=months[m][1])
    