# -*- coding: utf-8 -*-
"""
Check CI of spectrum assming random noise
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy import signal
from scipy import stats

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 14

#%% Inputs
T0=24#[h] fundamental period
DT=0.5#[h] sampling period
sigmas_noise=[1,5,10,20]# stdev of noise
M=1000#MC launches
p_value=0.05

#graphics
P_bins=np.arange(20)#psd bins

#%% Initialization
t=np.arange(0,T0*10,DT)
x=2**0.5*np.sin(t/T0*2*np.pi)
N=len(x)

#graphics
plt.figure(figsize=(18,8))
ctr=1

#%% Main
f,p0=signal.periodogram(x,fs=1/DT,scaling='density')
p=p0/np.var(x)*f**2
for sigma_noise in sigmas_noise:
    X=np.zeros((M,N))
    P=np.zeros((M,len(f)))
    T=np.zeros((M,len(f)))
    for m in range(M):
        X[m,:]=x+np.random.normal(0,sigma_noise,N)
        P[m,:]=signal.periodogram(X[m,:],fs=1/DT,scaling='density')[1]/np.var(X[m,:])*f**2
        T[m,:]=1/f
    P_sel=P[:,(f!=1/T0)*(f>0)].ravel()
    T_sel=T[:,(f!=1/T0)*(f>0)].ravel()
    x_var=np.mean(np.var(X,axis=1))
    
    prob=np.sum(P_sel>stats.chi2.ppf(1-p_value, 2)*DT/T_sel**2)/len(P_sel)
    
    #%% Plots
    plt.subplot(2,len(sigmas_noise),ctr)
    plt.loglog(1/f,P.T,'.k',alpha=0.1,markersize=1)
    plt.plot(1/f,stats.chi2.ppf(1-p_value, 2)*DT*f**2,'--r')
    plt.ylim([10**-6,10])
    plt.xlabel('$T$ [hours]')
    if ctr==1:
        plt.ylabel('$P_T$ [hours$^{-1}$]')
    plt.grid()
    plt.title(r'$\sigma_{noise}/\sigma_{clean}='+str(sigma_noise)+'$')
    
    plt.subplot(2,len(sigmas_noise),ctr+len(sigmas_noise))
    H=np.histogram(P_sel*T_sel**2/DT,density=True,bins=P_bins)[0]
    plt.plot(utl.mid(P_bins),H,'k',label='Data (noise only)')
    plt.plot(P_bins,stats.chi2.pdf(P_bins, 2),'--r',label=r'$\chi^2_2$')
    plt.text(2,0.5,'Probability of exceeding \n '+str(p_value)+' p-value='+str(np.round(prob*100,1))+'%')
    plt.xlabel('$P_T ~ T^2 ~ f_s$ [hours]')
    if ctr==1:
        plt.ylabel('Probability')
    plt.ylim([0,0.6])
    plt.grid()
    ctr+=1
plt.legend(draggable=True)