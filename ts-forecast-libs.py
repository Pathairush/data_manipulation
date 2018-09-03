
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages

import seaborn as sns
from scipy import stats
from __future__ import print_function # For compatibility for python 2 

import sys
print(sys.version)
#https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
plt.style.use('ggplot')
#plt.style.use('seaborn-muted')
plt.rcParams['figure.figsize'] = (10, 10)

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_formats = {'png', 'retina'}")


# In[20]:


# x is the pandas series 
def check_normality(x):
    T = len(x)

    fig = plt.figure(figsize=(14,8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.plot(x, color='blue', alpha=0.5, linestyle='--', marker='o', ms=4)
    ax1.set_title('Residuals')
    
    # Scatter plot
    ax2.scatter(x, x.shift())
    ax2.set_title(r'Scatter plot of $x_t$ vs $x_{t+1}$')
    
    # histogram plot
    sns.distplot(x, hist=True, kde=True, ax=ax3)
    ax3.set_title('Histogram of residiuals')

 
    # qq plot
    sm.qqplot(x,line='q',ax=ax4)
    ax4.set_title('Normal plot of residuals')
    plt.show()
    
    # ACF and PACF plots
    max_lag = min(20,int(0.8*T))
    fig = plt.figure(figsize=(14,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    plot_acf(x, lags=max_lag,ax=ax1); 
    # Draw 95% c.i. bounds
    ax1.hlines(stats.norm.isf(0.025)/np.sqrt(T),0,max_lag, linestyles='dashed',lw=1)
    ax1.hlines(-stats.norm.isf(0.025)/np.sqrt(T),0,max_lag, linestyles='dashed',lw=1)
    plot_pacf(x, lags=min(20,int(0.8*T)), ax=ax2); 
    plt.show()
    
    from statsmodels.stats.diagnostic import acorr_ljungbox
    h = min(10, int(len(x)/5))
    tstat,pval = acorr_ljungbox(x, lags=h)
    print('Results of Ljung-box Test:')
    print("Max lags: ", h)
    print("Test statistics:", tstat)
    print("P-values:", pval)


# In[ ]:


from sklearn.metrics import mean_squared_error
def get_forecast_accuracy(y, yhat):
    
    rmse = np.sqrt(mean_squared_error(y, yhat)) 
    et = y - yhat
    
    # Remove terms not a number (yt = 0)
    etyt = np.abs(et/y)
    etyt = etyt[np.isfinite(etyt)]
    mape = np.mean(etyt)*100
    wape = np.sum(np.abs(et))*100/np.sum(np.abs(y))
    
    print('RMSE: %.3f' % rmse)
    print('MAPE: %.2f%%' % mape)
    print('WAPE: %.2f%%' % wape)


# In[ ]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

def test_stationarity(timeseries):
    
    timeseries = timeseries[np.isfinite(timeseries)]
    T = len(timeseries)
    
    #Determing rolling statistics
    winsize = max(int(T/20),10)
    rolmean = timeseries.rolling(window=winsize,center=False).mean() 
    rolstd = timeseries.rolling(window=winsize,center=False).std()

    #Plot rolling statistics:
    plt.figure(figsize=(8,6))
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Augmented Dickey-Fuller test:
    print('Results of Augmented Dickey-Fuller Test:')
    dftest = adfuller(timeseries)
    dfoutput = pd.Series(dftest[0:2], index=['Test Statistic','p-value'])
    #for key,value in dftest[4].items():
    #    dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput);
    
    #Perform KPSS test:
    print('\nResults of KPSS Test:')
    dftest = kpss(timeseries, regression='ct') # Use 'c' for stationary around constant level with no trend.
    dfoutput = pd.Series(dftest[0:2], index=['Test Statistic','p-value'])
    print(dfoutput);


# In[15]:


def plot_ccf(x,y):
    T = len(x)
    maxlag = min(20, int(T/2))
    output = list()
    mx,sx = np.mean(x),np.std(x,ddof=1)
    my,sy = np.mean(y),np.std(y,ddof=1)
    
    H = np.arange(-maxlag,maxlag+1,1)
    x.index = y.index = np.arange(0,T)
    
    for h in H:
        xh,yh = x-mx, y-my
        if h >= 0:
            xh = xh[h:].values
            yh = yh[:T-h].values
        else:
            xh = xh[:T+h].values
            yh = yh[-h:].values
        output.append(np.sum(xh*yh)/T) 
    
    rxy = np.array(output)/(sx*sy)

    plt.figure(figsize=(8,4));
    plt.stem(H, rxy, markerfmt=' ')
    plt.hlines(stats.norm.isf(0.025)/np.sqrt(T),-maxlag,maxlag, linestyles='dashed',lw=1)
    plt.hlines(-stats.norm.isf(0.025)/np.sqrt(T),-maxlag,maxlag, linestyles='dashed',lw=1)
    plt.ylim([min(-0.2, min(rxy)*1.2), max(rxy)*1.2])
    plt.xlabel("lag h")


# In[ ]:


#from statsmodels.stats.diagnostic import acorr_breusch_godfrey
#bg_test = acorr_breusch_godfrey(reg_mdl, nlags=5)
#print('Results of Breusch-Godfrey Test for auto-correlation:')
#print('F-statistic:', bg_test[2])
#print('P-value:', bg_test[3])

