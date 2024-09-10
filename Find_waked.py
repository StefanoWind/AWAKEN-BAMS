# -*- coding: utf-8 -*-
'''
Assign waked flag and distance from wake based on IEC 61400-12-1 (2017)
'''
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from datetime import datetime

#%% Inputs
source_lyt=os.path.join(cd, 'data/20231026_AWAKEN_layout.nc')
WD=np.arange(0,360,1)#[deg] set of wind directions

#graphics
site_plot='A1'

#%% Initialization
TRB=xr.open_dataset(source_lyt,group='Turbines')
SIT=xr.open_dataset(source_lyt,group='Ground sites')

#site
D=TRB['Diameter'].values
D[np.isnan(D)]=np.nanmax(D)
xT=TRB['x UTM'].values
yT=TRB['y UTM'].values

#zeroing
waked=np.zeros((len(SIT['Site name']),len(WD)))

#%% Main
i_s=0
for s in SIT['Site name'].values:
    i_wd=0
    for wd in WD:
        x0=SIT['x UTM'].sel({'Site name': s}).values
        y0=SIT['y UTM'].sel({'Site name': s}).values
        
        L,th0=utl.cart2pol(xT-x0, yT-y0)#relative distance and orientation
        
        wd_align=(90-th0)%360#wind direction of maximum wake interaction
        wd_diff=np.abs(utl.ang_diff(wd_align,wd))#different from maximum wake direction
        waked_sector=1.3*utl.arctand(2.5*D/L+0.15)+10#[IEC 61400-12-1:2017] waked sector
        waking=wd_diff<waked_sector/2
        
        if np.sum(waking)>0:
            waked[i_s,i_wd]=np.nanmin(L[waking]/D[waking])
        else:
            waked[i_s,i_wd]=-9999
        
        if s==site_plot:
            plt.figure()
            
            plt.plot(xT-x0,yT-y0,'ok',label='Non-waking turbines',markersize=2)
            plt.plot(xT[waking]-x0,yT[waking]-y0,'or',label='Waking turbines',markersize=2)
            plt.plot(0,0,'^r',markersize=10)
            if np.sum(waking)>0:
                plt.title(r'Distance from closest waking turbine: '+str(int(waked[i_s,i_wd]))+'$D$')
            else:
                plt.title('No waking turbines')
            plt.xlim([-40000,+40000])
            plt.ylim([-40000,+40000])
            plt.xlabel('W-E [m]')
            plt.ylabel('S-N [m]')
            ax=plt.gca()
            xx=utl.cosd(270-wd)*3000
            yy=utl.sind(270-wd)*3000
            ax.arrow(0, -10000, xx, yy, head_width=30, head_length=30, fc='b', ec='b',linewidth=4)
            utl.axis_equal()
            plt.legend()
            plt.grid()
            
            utl.mkdir(os.path.join(cd,'figures',s))
            plt.savefig(os.path.join(cd,'figures/',s,'{:03d}'.format(wd)+'deg.png'))
            plt.close()
        
        i_wd+=1
    i_s+=1

#%% Output
Output = xr.Dataset({
'waked': xr.DataArray(
            data   = waked,   # enter data here
            dims   = ['site','wind_direction'],
            coords = {'site': SIT['Site_name'].values,'wind_direction':WD},
            attrs  = {'_FillValue': -9999,'description':'Distance of the closest waking turbine'})})

Output.to_netcdf('data/'+datetime.strftime(datetime.now(),'%Y%m%d')+'_AWAKEN_waked.nc')