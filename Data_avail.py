# -*- coding: utf-8 -*-
"""
Plot data availability form DAP
"""

import os
cd=os.getcwd()
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/Custom_functions/dap-py') 
from doe_dap_dl import DAP
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

sdate='20230101'#start date
edate='20231231'#end date

channels={'B':'awaken/sb.assist.z01.c0'}#channels names

formats={'B':'nc'}#format

ext={'B':''}#extension (put '' if not applicable)

#graphics
time_res=3600*24#[s]

#%% Initialization
with open(source_config, 'r') as fid:
    config = yaml.safe_load(fid)
    
a2e = DAP('a2e.energy.gov',confirm_downloads=False)
a2e.setup_cert_auth(username=config['password'], password=config['username'])

#%% Main
plt.figure(figsize=(18,len(channels)))
ctr=1
for s in channels:
    
    if ext[s]!='':
        _filter = {'Dataset': channels[s],
                    'date_time': {
                        'between': [sdate+'000000',edate+'000000']
                    },
                    'file_type':formats[s],
                    'ext':ext[s]
                }
    else:
        _filter = {'Dataset': channels[s],
                    'date_time': {
                        'between': [sdate+'000000',edate+'000000']
                    },
                    'file_type':formats[s],
                }
    
    files=a2e.search(_filter)

    dates=[]
    size=0
    for f in files:
        dates=np.append(dates,datetime.strptime(f['data_date'],'%Y%m%d'))
        size+=f['size']/10**9
        
    ax=plt.subplot(len(channels),1,ctr)

    for d in dates:
        ax.fill_between([d,d+timedelta(seconds=time_res)],y1=[1,1],y2=ctr*2,color='g')
    plt.title(channels[s]+f': {len(files)} total files, {np.round(size,2)} Gb')
    
    plt.xlim([datetime.strptime(sdate, '%Y%m%d'),datetime.strptime(edate, '%Y%m%d')])
    plt.yticks([])
    ctr+=1
     
plt.tight_layout()