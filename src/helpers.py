# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 20:58:38 2018

"""

import pandas as pd
import numpy as np
from arch import arch_model
from scipy.stats import norm
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
#from pytrends.request import TrendReq
#%%

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
    
def calc_imp_vol(premium, option_params):
    '''
    Calculates the BS implied volatility given an option's premium and
    underlying parameters. Assumes that the highest value of value can be
    200%
    '''
    def option_loss(sigma, option_params, premium):
        S, K, r, y, T, call_flag = option_params
        bs_price = black_scholes_pricer(S, K, r, y, T, sigma, call_flag)
        #loss = (premium - bs_price)**2*1E6
        loss = (premium - bs_price)**2
        #print('Sigma:', sigma, 'Loss:', loss)
        return loss
    
    sigma_guess = 0.15
    eps = np.finfo(float).eps
    # Constraint sigma to be in (eps, 1)
    bounds = ((eps, 2.),)
    opt_res = minimize(fun=option_loss,
                       x0=sigma_guess,
                       args=(option_params, premium),
                       #method='SLSQP',
                       bounds=bounds,
                       #options={'maxiter':int(1E3), 'disp':True}
                       options={'maxiter':int(1E3)}
                      )
    sigma_imp = opt_res.x[0]
    return sigma_imp

def options_implied_vol_data_clean(data_df):
    '''
    Clean the options' implied vol data
    '''
    columns_to_keep = ['date', 'exdate', 'last_date', 'cp_flag',
                       'strike_price', 'best_bid', 'best_offer',
                       'impl_volatility', 'delta', 'optionid']
    data_df = data_df[columns_to_keep]
    # Convert date columns to pandas datetime
    datetype_columns = ['date', 'exdate', 'last_date']
    for date_col in datetype_columns:
        data_df[date_col] = pd.to_datetime(data_df[date_col], yearfirst=True,
                                           format='%Y%m%d')
    # Adjust the strike price because it's multiplied by 1000
    data_df['strike_price'] = data_df['strike_price']/1000
    data_df['mid_price'] = (data_df['best_bid'] + data_df['best_offer'])/2
    data_df['T'] = (data_df['exdate'] - data_df['date']).dt.days
    return data_df

#%%
###############################################################################
## Scrapers
###############################################################################        
        
def google_trends(keyword_list=["Blockchain"],cat= 0,
                  time_frame ="2008-01-01 2017-12-01",
                  gprop = "",make_plot=True):
    '''
    Downloads Time serries data for the keywords.Make sure you have the library:
    pip install pytrends
    
    examples: https://mancap314.github.io/googletrends-py.html
    
    Parameters:
    keyword list: Give combination for google search combination
    time_frame: Give in "yyyy-mm-dd yyyy-mm-dd"  fixed for now.
    cat: 1138 for business and finance for more: 
    https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories
    gprop: can be 'images', 'news', 'youtube' or 'froogle'
    make_plot: Set False to mute the popping graphs.
    
    Returns: Returns a Data Frame
    Example:
    test_df = google_trends(keyword_list= ["Bulls","Bears"],
                        gprop = "news")
    
    '''
    from pytrends.request import TrendReq        
    pytrends = TrendReq(hl='en-US',tz=360)
    #kw_list = ["Blockchain"]
    pytrends.build_payload(kw_list=keyword_list,cat=cat, timeframe=time_frame,
                           geo='', gprop='')
    df=pytrends.interest_over_time()
    if make_plot ==False:
        print("Your Download looks like:")
        df.plot.line()
    else: 
        print("Download for ",keyword_list, "completed")
    return(df)

