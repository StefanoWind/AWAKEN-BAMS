# -*- coding: utf-8 -*-
"""
Print data availability
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18

#%% Inputs
#dataset
sources=['data/20230101.000500-20240101.224500.awaken.glob.summary.csv',
             'data/20230101.000500-20240101.224500.awaken.sa1.summary.csv',
             'data/20230207.203500-20230911.121500.awaken.sa2.summary.csv',
             'data/20230101.000500-20240101.224500.awaken.sh.summary.csv']

max_TKE=10#[m^2/s^2] maximum tke

#%% Main
for s in sources:
    LOG=pd.read_csv(os.path.join(cd,s))
    LOG['Rotor-averaged TKE [m^2/s^2]']=LOG['Rotor-averaged TKE [m^2/s^2]'].where(LOG['Rotor-averaged TKE [m^2/s^2]']>0).where(LOG['Rotor-averaged TKE [m^2/s^2]']<max_TKE)
    print(s)
    print(LOG['UTC Time'].values[0])
    print(LOG['UTC Time'].values[-1])
    print(np.sum(~np.isnan(LOG['Hub-height wind speed [m/s]'])))
    print(np.sum(~np.isnan(LOG['Rotor-averaged TKE [m^2/s^2]'])))