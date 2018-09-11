# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 20:58:38 2018

"""

import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import norm
import os

# Change the below to local directory
os.chdir("C:\\Users\\Saurabh\\Documents\\Python Scripts\\Fall\\230T\\Project")

implied_vol = pd.read_csv(".//data//implied_vol.csv")

# garch_model = arch_model(y=implied_vol["impl_volatility"],
#                         mean="Constant", p=1, q=1)
# model_result = garch_model.fit()


def fit_garch_model(ts=implied_vol["impl_volatility"], p=1, q=1):
    ''' Takes in the time series returns the parameters. Default params are p=1
    and q=1 '''
    garch_model = arch_model(y=ts, mean="Constant",
                             p=1, q=1)
    model_result = garch_model.fit()
    # params = model_result.params
    return(model_result)

def kernel_smoothing():
    ''' Place holder for Nathan's Kernel smoothing to extract the smooth part
    from Variance time series'''
    
def empirical_smoothing():
    ''''Place holder for Salman to add Empirical Smoothing Function based on 
    230E Slides'''
    
def backtesting_algo():
    '''Place holder for Amrit to add Backtesting Function. Could be split into 
    other functions. Should take in 
    a. Price time-series.
    b. Signal time-series.
    <Modify as per requirement>     
    '''
    
def black_scholes_pricer(S, K, r, y, T, sigma, call_flag=False):
    '''
    Black-Scholes pricer of a European Option
    '''
    phi = 1 if call_flag else -1
    x_p = (np.log(S/K) + (r-y)*T)/(sigma*np.sqrt(T)) + sigma*np.sqrt(T)/2
    x_m = x_p - sigma*np.sqrt(T)
    N_x_p = norm.cdf(phi*x_p, loc=0, scale=1)
    N_x_m = norm.cdf(phi*x_m, loc=0, scale=1)
    return phi*(S*np.exp(-y*T)*N_x_p - K*np.exp(-r*T)*N_x_m)

def convert_prob_forecast_vol(forecast_prob, r, thresh=0.1, delta_t=7/365):
    '''
    Convert forecasted probability of stock log returns exceeding a threshold
    to implied vol forecast
    '''
    numerator = thresh - r*delta_t
    forecast_prob = forecast_prob if numerator >= 0 else 1 - forecast_prob
    denominator = np.sqrt(delta_t)*norm.ppf(1 - forecast_prob, loc=0, scale=1)
    if denominator == 0:
        return float('inf')
    else:
        return np.abs(numerator)/denominator

#%%