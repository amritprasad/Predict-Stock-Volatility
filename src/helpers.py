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
    # Constraint sigma to be in (eps, 3)
    bounds = ((eps, 3.),)
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
    # Convert implied vol to decimals from %
    data_df['impl_volatility'] = data_df['impl_volatility']/100
    # Engineer new helper columns
    data_df['mid_price'] = (data_df['best_bid'] + data_df['best_offer'])/2
    data_df['T'] = (data_df['exdate'] - data_df['date']).dt.days
    # Drop redundant columns
    redundant_columns = ['exdate']
    data_df.drop(redundant_columns, inplace=True, axis=1)
    return data_df

def combine_data(options_data_df, stock_data_df):
    '''
    Extracts the stock price and dividend yield for the underlying stock.
    Combines with the options' data.
    '''
    #stock_data_df = spx_data_df.copy()
    stock_columns = ['Dates', 'IDX_EST_DVD_YLD', 'PX_LAST']
    stock_data_df = stock_data_df[stock_columns]
    stock_data_df.rename(columns={'Dates':'date', 'IDX_EST_DVD_YLD': 'y',
                                  'PX_LAST': 'S'}, inplace=True)
    options_data_df = options_data_df.merge(stock_data_df, on='date',
                                            how='inner')
    return options_data_df

def trade_best_option(date, forecast_imp_vol, data_df, look_ahead=7,
                      long_only=False, direction=None):
    '''
    Selects the best option to trade on the basis of the forecasted implied
    volatility and the direction of stock price move. Provides the PnL after
    exactly 7 days of holding the option.
    Inputs Types:
        1) date: pandas._libs.tslibs.timestamps.Timestamp
        2) forecast_imp_vol: float
        3) data_df: pandas.core.frame.DataFrame
        4) look_ahead: int (forecast window)
        5) long_only: bool
        6) direction: int (+/- 1)
    '''
    if direction is not None:
        assert direction in [-1, 1], 'Stock direction can only be +/- 1'
        option_type = 'C' if direction == 1 else 'P'
    date_fwd = date + pd.Timedelta(days=look_ahead)
    count = 1
    if np.logical_not(any(data_df['date'] == date_fwd)):
        date_fwd = date + pd.Timedelta(days=look_ahead+count)
        count += 1
    flter_ind = (data_df['date'] == date) | (data_df['date'] == date_fwd)
    if long_only:
        flter_ind = flter_ind & (data_df['cp_flag'] == option_type)
    best_options_df = data_df[flter_ind]
    # Remove non-liquid options
    best_options_df = best_options_df[best_options_df['n_days_lt'] == 0]
    id_groupby = best_options_df.groupby(['optionid'])
    liquid_options_list = id_groupby['impl_volatility'].count() == 2
    liquid_options_list = list(liquid_options_list.index[liquid_options_list])
    liquid_filter_ind = best_options_df['optionid'].isin(liquid_options_list)
    best_options_df = best_options_df[liquid_filter_ind]    
    # Calculate PnL of executing trade on the best option
    output_dict = {'Unwind_Date': date_fwd, 'PnL': np.nan}
    if best_options_df.shape[0] != 0:        
        cur_date_df = best_options_df[best_options_df['date'] == date]
        fwd_date_df = best_options_df[best_options_df['date'] == date_fwd]
        cur_date_df['abs_vol_diff'] = np.abs(cur_date_df['impl_volatility']
                                             - forecast_imp_vol)
        cur_date_df = cur_date_df.sort_values(by='abs_vol_diff',
                                              ascending=False)
        trade_option_id = cur_date_df['optionid'].iloc[0]
        # Price current best option
        S = cur_date_df['S'].iloc[0]
        K = cur_date_df['strike_price'].iloc[0]
        r = cur_date_df['r'].iloc[0]
        y = cur_date_df['y'].iloc[0]
        T = cur_date_df['T'].iloc[0]/365
        sigma_impl = cur_date_df['impl_volatility'].iloc[0]
        call_flag = cur_date_df['cp_flag'].iloc[0] == 'C'
        cur_price = black_scholes_pricer(S, K, r, y, T, sigma_impl, call_flag)
        # Extract parameters of the option for remaining days
        # Implement continuous delta-hedge strategy: To Do
        # Price best option 7 days ahead
        fwd_option_ind = fwd_date_df['optionid'] == trade_option_id
        fwd_date_df = fwd_date_df[fwd_option_ind]
        S_fwd = fwd_date_df['S'].iloc[0]
        r_fwd = fwd_date_df['r'].iloc[0]
        y_fwd = fwd_date_df['y'].iloc[0]
        T_fwd = fwd_date_df['T'].iloc[0]/365
        sigma_impl_fwd = fwd_date_df['impl_volatility'].iloc[0]
        fwd_price = black_scholes_pricer(S_fwd, K, r_fwd, y_fwd, T_fwd,
                                         sigma_impl_fwd, call_flag)
        pnl = fwd_price - cur_price
        vol_diff = sigma_impl - forecast_imp_vol
        # Implement directional strategy
        if direction is not None:
            if np.logical_not(long_only):            
                # Sell Direction and best option is call
                if (direction == -1) & call_flag:
                    pnl = -pnl
                # Buy Direction and best option is put
                elif (direction == 1) & np.logical_not(call_flag):
                    pnl = -pnl
        # Implement vol strategy            
        else:
            # Short overpriced options
            if vol_diff > 0:
                pnl = -pnl
        
        output_dict['PnL'] = pnl
        
    return output_dict
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

