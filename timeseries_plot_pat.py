# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:49:33 2018

@author: pathairs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import autocorrelation_plot
plt.rc({'font.size': 18})


def ts_groupby_plot(ts,resample='y'):
    grouped = ts.to_frame().resample(resample)
    for key,values in grouped.groups.items():
        grouped.get_group(key).plot()

def ts_hist(ts,figsize=(16,9)):
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1);
    ts.hist(grid=False)
    plt.title("histogram of ts")
    plt.subplot(1,2,2);
    ts.plot('kde')
    plt.title("kernel density plot of ts")

def ts_boxplot_overtime(ts,resample='y',selected_col=None):
    grouped = ts.resample(resample)[selected_col]
    df_overtime = pd.DataFrame()
    for key,values in grouped.groups.items():
    #    print(key)
        try:
            tempdf = grouped.get_group(key).to_frame()
        except:
            print("error in get group variable")
            
        if resample == 'y':
            tempdf.columns = pd.Index([tempdf.index.year[0].astype('str')])
        if resample == 'm':
            tempdf.columns = pd.Index([tempdf.index.year[0].astype('str') + "_" + tempdf.index.month[0].astype('str')])  
        if resample == 'd':
            tempdf.columns = pd.Index([tempdf.index.day[0].astype('str')])  
        try:   
            df_overtime = pd.concat([df_overtime,tempdf],sort=False)
        except:
            print("error in concat variable")
        del tempdf
    df_overtime.boxplot(sym='k.')
    

def ts_lag_plot(ts,title="lag plot"):   
    lag_plot(ts)
    plt.title(title)


def ts_acc_plot(ts,title="autocorrelation plot"):   
    autocorrelation_plot(ts)
    plt.title(title)

