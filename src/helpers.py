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
from arch.univariate import GARCH,ARX
from arch import arch_model
#from pytrends.request import TrendReq
#%%

def fit_garch_model(ts, p=1, q=1):
    ''' Takes in the time series returns
    returns the parameters. Default params are p=1
    and q=1 '''
    garch_model = arch_model(y=ts, mean="HAR", lags=[1], vol="garch",
                             p=p, q=q)
    #garch_model = arch_model(y=ts, vol='Garch', p=p, o=0, q=q, dist='Normal')
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
    
def black_scholes_pricer(S, K, r, y, T, sigma, call_flag=False,
                         delta_flag=False):
    '''
    Black-Scholes pricer of a European Option. Will return delta also in case
    delta_flag is True
    '''
    phi = 1 if call_flag else -1
    x_p = (np.log(S/K) + (r-y)*T)/(sigma*np.sqrt(T)) + sigma*np.sqrt(T)/2
    x_m = x_p - sigma*np.sqrt(T)
    N_x_p = norm.cdf(phi*x_p, loc=0, scale=1)
    N_x_m = norm.cdf(phi*x_m, loc=0, scale=1)
    out = phi*(S*np.exp(-y*T)*N_x_p - K*np.exp(-r*T)*N_x_m)
    if delta_flag:
        out = phi*np.exp(-y*T)*N_x_p
    return out

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
        cur_date_df['abs_vol_diff'] = np.abs(cur_date_df['impl_volatility']
                                             - forecast_imp_vol)
        cur_date_df = cur_date_df.sort_values(by='abs_vol_diff',
                                              ascending=False)
        trade_option_id = cur_date_df['optionid'].iloc[0]
        # Extract parameters of the option for all days
        K = cur_date_df['strike_price'].iloc[0]
        call_flag = cur_date_df['cp_flag'].iloc[0] == 'C'
        strategy_filter_ind = (data_df['date'] >= date) & (data_df['date'] <= date_fwd)
        strategy_filter_ind = strategy_filter_ind & (data_df['optionid'] == trade_option_id)
        S_arr = data_df['S'][strategy_filter_ind].ffill()
        r_arr = data_df['r'][strategy_filter_ind].ffill()
        y_arr = data_df['y'][strategy_filter_ind].ffill()
        T_arr = data_df['T'][strategy_filter_ind].ffill()/365
        sigma_impl_arr = data_df['impl_volatility'][strategy_filter_ind].ffill()
        bs_pricer = np.vectorize(black_scholes_pricer)
        option_price_arr = bs_pricer(S_arr, K, r_arr, y_arr, T_arr, sigma_impl_arr, call_flag)
        #delta_arr = np.abs(bs_pricer(S_arr, K, r_arr, y_arr, T_arr, sigma_impl_arr, call_flag, True))
        delta_arr = np.abs(bs_pricer(S_arr, K, r_arr, y_arr, T_arr, forecast_imp_vol, call_flag, True))
        # Implement continuous delta-hedge strategy
        vol_diff = sigma_impl_arr.iloc[0] - forecast_imp_vol
        phi = 1 if call_flag else -1
        account = option_price_arr[0] - delta_arr[0]*S_arr.iloc[0]*phi        
        for idx in range(1, len(option_price_arr)-1):
            turnover_pnl = phi*(delta_arr[idx-1] - delta_arr[idx])*S_arr.iloc[idx]
            #print(turnover_pnl)
            account += turnover_pnl
        account -= option_price_arr[len(option_price_arr)-1]
        account += phi*delta_arr[-2]*S_arr.iloc[-1]        
        # Short options' strategy if vol diff > 0
        pnl = account if vol_diff > 0 else -account
        
        output_dict['PnL'] = pnl
        
    return output_dict
#%%
###############################################################################
## Scrapers
###############################################################################        
        
def google_trends(keyword_list=["Blockchain"],cat= 0,
                  time_frame ="2000-01-01 2017-12-01",
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


def scrape_these_words(key_words = ["bears","bulls"],path = "../data",
                       file_name = "positive_words.csv"):
    
    file_path = path + "/" + file_name 
    if os.path.isfile(file_path):
        print("File Found in path. Reading it to append")
        this_df = pd.read_csv(file_path)
        this_df["date"] = pd.to_datetime(this_df["date"])
        this_df.set_index("date",inplace = True)
        existing_words = list(this_df.columns)
        count = 1
    else: 
        print("Creating new file and df")
        count = 0
        existing_words = []
        
    for this_word in key_words:
        if this_word in existing_words:
            print("Word: ",this_word," already exists")
            continue
        
        #this_word = key_words[1]
        temp_df= google_trends([this_word])
        if(count == 0):
            this_df = temp_df[[this_word]]
            count = count +1
        else: 
            this_df = pd.merge(this_df,temp_df[[this_word]],left_index=True,
                               right_index=True,how = "outer")
    this_df.to_csv(file_path)
            
     