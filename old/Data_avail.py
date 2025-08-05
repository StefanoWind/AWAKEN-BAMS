# -*- coding: utf-8 -*-
"""
Plot data availability form WDH
"""

import os
cd=os.getcwd()
import sys
from datetime import datetime
from datetime import timedelta
import numpy as np
from matplotlib import pyplot as plt
import yaml
import matplotlib
import warnings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 12

plt.close('all')
warnings.filterwarnings('ignore')

#%% Inputs
source_config=os.path.join(cd,'config.yaml')

#WDH info
sdate='20230101'#[%Y%m%d] start date
edate='20231231'#[%Y%m%d] end date

#channel names
channels={'A1':'awaken/sa1.lidar.z03.c1',
          'A2':'awaken/sa2.lidar.z01.c1',
          'H':'awaken/sh.lidar.z02.c1'}

#file formats
formats={'A1':'nc',
         'A2':'nc',
         'H':'nc'}

#extension e.g. user1 or rhi or assistsummary (put '' if not applicable)
ext={'A1':'',
    'A2':'',
    'H':''}

MFA=False#use multi-factor authentication (for CRADA-protected channels)

#graphics
time_res=3600*24#[s] time duration of one file in the timeline

#%% Functions
def dap_search(channel,sdate,edate,file_type,ext1,time_search=30):
    '''
    Wrapper for a2e.search to avoid timeout:
        Inputs: channel name, start date, end date, file format, extention in WDH name, number of days scanned at each loop
        Outputs: list of files mathing the criteria
    '''
    dates_num=np.arange(utl.datenum(sdate,'%Y%m%d%H%M%S'),utl.datenum(edate,'%Y%m%d%H%M%S'),time_search*24*3600)
    dates=[utl.datestr(d,'%Y%m%d%H%M%S') for d in dates_num]+[edate]
    search_all=[]
    for d1,d2 in zip(dates[:-1],dates[1:]):
        
        if ext1!='':
            _filter = {
                'Dataset': channel,
                'date_time': {
                    'between': [d1,d2]
                },
                'file_type': file_type,
                'ext1':ext1, 
            }
        else:
            _filter = {
                'Dataset': channel,
                'date_time': {
                    'between': [d1,d2]
                },
                'file_type': file_type,
            }
        
        search=a2e.search(_filter)
        
        if search is None:
            print('Invalid authentication')
            return None
        else:
            search_all+=search
    
    return search_all

#%% Initialization
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
#imports
sys.path.append(config['path_utils'])
sys.path.append(config['path_dap'])
import utils as utl
from doe_dap_dl import DAP
    
#WDH setup
a2e = DAP('a2e.energy.gov',confirm_downloads=False)
if MFA==False:
    a2e.setup_cert_auth(username=config['username'], password=config['password'])
else:
    a2e.setup_two_factor_auth(username=config['username'], password=config['password'])

#%% Main
plt.figure(figsize=(18,len(channels)))
ctr=1
for s in channels:
    
    files=dap_search(channels[s],sdate+'000000',edate+'000000',formats[s],ext[s])#search files

    #extract info
    dates=[]
    size=0
    for f in files:
        dates=np.append(dates,datetime.strptime(f['date_time'],'%Y%m%d%H%M%S'))
        size+=f['size']/10**9
   
    #plot
    ax=plt.subplot(len(channels),1,ctr)     
    for d in dates:
        ax.fill_between([d,d+timedelta(seconds=time_res)],y1=[1,1],y2=0,color='g')
    plt.title(channels[s]+f': {len(files)} total files, {np.round(size,2)} Gb')
    
    plt.xlim([datetime.strptime(sdate, '%Y%m%d'),datetime.strptime(edate, '%Y%m%d')])
    plt.yticks([])
    ctr+=1
     
plt.tight_layout()