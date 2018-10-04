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
#from arch.univariate import GARCH,ARX
import bisect
import time
from scipy.stats import percentileofscore
#from pytrends.request import TrendReq
#%%

def fit_garch_model(ts, p=1, q=1):
    ''' Takes in the time series returns
    returns the parameters. Default params are p=1
    and q=1 '''
    garch_model = arch_model(y=ts, vol="garch", p=p, q=q)
    #garch_model = arch_model(y=ts, vol='garch', p=p, o=0, q=q)
    model_result = garch_model.fit()
    #params = model_result.params
    return(model_result)
    
def forecast_garch(fitted_result, spx_pred_data, init_resid, init_vol):
    '''
    Produce GARCH results. Input is the fitted model.
    '''
    #spx_pred_data = spx_data[train_idx:cv_idx].copy()
    mu, omega, alpha, beta = fitted_result.params
    returns_pred_series = spx_pred_data['Returns'].values
    returns_pred_resid = returns_pred_series - mu
    pred_variance = np.array([np.nan]*len(returns_pred_series))
    pred_variance[0] = omega + alpha*init_resid**2 + beta*init_vol**2
    for i in range(1, len(pred_variance)):
        pred_variance[i] = omega + alpha*returns_pred_resid[i-1]**2
        pred_variance[i] += beta*pred_variance[i-1]
    
    pred_vol = np.sqrt(pred_variance)
    
    return pred_vol

def forecast_nn(garch_fitted_result, init_resid, init_vol, nn_innovations):
    '''
    Produce NN results using the GARCH results
    '''
    mu, omega, alpha, beta = garch_fitted_result.params
    pred_variance = np.array([np.nan]*len(nn_innovations))
    pred_variance[0] = omega + alpha*init_resid**2 + beta*init_vol**2
    for i in range(1, len(pred_variance)):
        pred_variance[i] = omega + alpha*nn_innovations[i-1]
        pred_variance[i] += beta*pred_variance[i-1]
    
    pred_vol = np.sqrt(pred_variance)
    
    return pred_vol

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
    
def calculate_ewma_vol(series, lmbda, window):
    '''
    Calculate volatility using Exponentially smoothed returns. Inputs-
    1) series: returns' series
    2) lmbda: decay parameter
    3) window: window of returns' taken for vol extimation
    '''
    #series, lmbda, window = spx_data["Returns"], 0.94, 63
    ewma_returns_series = series.ewm(alpha=1-lmbda, min_periods=window).mean()
    ewma_vol_series = ewma_returns_series.rolling(window).std()
    #ewma_vol_series = ewma_vol_series[~np.isnan(ewma_vol_series)]
    
    return ewma_vol_series
    
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
    # Engineer new helper columns
    data_df['mid_price'] = (data_df['best_bid'] + data_df['best_offer'])/2
    data_df['T'] = (data_df['exdate'] - data_df['date']).dt.days
    data_df['n_days_lt'] = (data_df['date'] - data_df['last_date']).dt.days
    # Drop redundant columns
    redundant_columns = ['exdate', 'last_date']
    data_df.drop(redundant_columns, inplace=True, axis=1)
    return data_df

def combine_data(options_data_df, stock_data_df):
    '''
    Extracts the stock price and dividend yield for the underlying stock.
    Combines with the options' data.
    '''
    #options_data_df, stock_data_df = options_implied_vol_df.copy(), bbg_data_df.copy()
    stock_columns = ['Dates', 'IDX_EST_DVD_YLD', 'PX_LAST',
                     'USSOC CMPN Curncy']
    stock_data_df = stock_data_df[stock_columns]
    stock_data_df.rename(columns={'Dates':'date', 'IDX_EST_DVD_YLD':'y',
                                  'PX_LAST':'S', 'USSOC CMPN Curncy':'r'},
                         inplace=True)
    # Convert columns to decimals
    stock_data_df['r'] = stock_data_df['r']/100
    stock_data_df['y'] = stock_data_df['y']/100
    # Forward fill prices
    stock_data_df.fillna(method='ffill', inplace=True)
    options_data_df = options_data_df.merge(stock_data_df, on='date',
                                            how='inner')
    return options_data_df

def trade_best_option(date, forecast_imp_vol, data_df, look_ahead=7,
                      long_only=False, direction=None, multiple=False,
                      atm_only=False, trade_expiry=True):
    '''
    Selects the best option to trade on the basis of the forecasted implied
    volatility and the direction of stock price move. Provides the PnL after
    exactly 7 days of holding the option.
    Inputs Types:
        1) date: pandas._libs.tslibs.timestamps.Timestamp
        2) forecast_imp_vol: float (assumed to be daily vol)
        3) data_df: pandas.core.frame.DataFrame
        4) look_ahead: int (forecast window)
        5) long_only: bool
        6) direction: int (+/- 1)
        7) multiple: bool - take multiple options' positions depending upon vol
                            diff
        8) atm_only: bool - take positions in near ATM options only
        9) trade_expiry: bool - trade options 7 days before expiry
    '''
    #date, forecast_imp_vol, data_df, look_ahead, long_only, direction = options_implied_vol_df['date'][34], 0.25, options_implied_vol_df.copy(), 7, False, None
    if direction is not None:
        assert direction in [-1, 1], 'Stock direction can only be +/- 1'
        option_type = 'C' if direction == 1 else 'P'
    date_fwd = date + pd.Timedelta(days=look_ahead)
    count = 1
    if np.logical_not(any(data_df['date'] == date_fwd)):
        date_fwd = date + pd.Timedelta(days=look_ahead+count)
        count += 1
    output_dict = {'PnL': np.nan, 'Trade_Type': np.nan,
                   'Option_Type': np.nan, 'Implied_Vol': np.nan}
    if trade_expiry:
        expiry_dates = data_df.loc[data_df['T'] == 0, 'date'].unique()
        if np.datetime64(date_fwd) not in expiry_dates:
            return output_dict
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
    # Choose options with smallest time to maturity >= look_ahead. Vol scaling
    # approximation would work best that way
    time_maturity = sorted(best_options_df.loc[best_options_df['date']==date,
                                               'T'].unique())
    best_T = bisect.bisect_left(time_maturity, look_ahead)
    if len(time_maturity) == 0:
        with open("no_liquid_options_dates.txt", "a") as myfile:
            date = pd.to_datetime(date)
            myfile.write('{:%Y/%m/%d}\n'.format(date))
        #print('{:%Y/%m/%d} has no liquid options!'.format(date))
        return output_dict
    best_T = time_maturity[best_T]    
    # Scale daily forecast vol by the appropriate number
    forecast_imp_vol = forecast_imp_vol*np.sqrt(252)
    #forecast_imp_vol = forecast_imp_vol*np.sqrt(best_T)
    time_filter_ind = (best_options_df['date'] == date)
    time_filter_ind = time_filter_ind & (best_options_df['T'] == best_T)
    valid_option_ids = best_options_df.loc[time_filter_ind, 'optionid']
    valid_option_ids = np.unique(valid_option_ids)
    time_filter_ind = best_options_df['optionid'].isin(valid_option_ids)
    best_options_df = best_options_df[time_filter_ind]
    # If only trading ATM options, remove the remaining
    if atm_only & (best_options_df.shape[0] != 0):
        del valid_option_ids
        atm_idx = (np.abs(best_options_df['delta'] - 0.5) < 0.1)
        atm_idx = atm_idx & (best_options_df['date'] == date)
        valid_option_ids = best_options_df.loc[atm_idx, 'optionid']
        valid_option_ids = np.unique(valid_option_ids)
        atm_idx = best_options_df['optionid'].isin(valid_option_ids)
        best_options_df = best_options_df[atm_idx]
    # Calculate PnL of executing trade on the best option
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
        option_price_arr = bs_pricer(S_arr, K, r_arr, y_arr, T_arr,
                                     sigma_impl_arr, call_flag)
        delta_arr = np.abs(bs_pricer(S_arr, K, r_arr, y_arr, T_arr,
                                     sigma_impl_arr, call_flag, True))
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
        # Assume multiple positions if true
        mult_factor = np.abs(vol_diff)/np.sqrt(best_T)/0.009 if multiple else 1
        mult_factor = max(mult_factor, 1)
        output_dict['PnL'] = pnl*mult_factor
        output_dict['Trade_Type'] = -mult_factor if vol_diff > 0 else mult_factor
        output_dict['Option_Type'] = cur_date_df['cp_flag'].iloc[0]
        output_dict['Implied_Vol'] = cur_date_df['impl_volatility'].iloc[0]
        
    return output_dict

def backtester(model_df, options_implied_vol_df, plot_title, look_ahead=7,
               long_only=False, direction=None, atm_only=False,
               trade_expiry=True):
    '''
    Calculates the total PnL and graphs the performance of the forecasts.
    Inputs-
        1) model_df: DataFrame - Columns = ['Forecast_Vol'], Index = 'Dates'
        2) options_implied_vol_df: DataFrame
    '''
    #model_df, plot_title = forecast_df.copy(), 'GARCH Back Test'
    pnl_series = [np.nan]*model_df.shape[0]
    options_traded = [np.nan]*model_df.shape[0]
    option_type = [np.nan]*model_df.shape[0]
    option_imp_vol = [np.nan]*model_df.shape[0]
    dates = np.array(model_df.index.get_level_values(0))
    forecast_vol_arr = model_df['Forecast_Vol'].values
    start = time.time()
    for count in range(model_df.shape[0]):
        cur_date = dates[count]
        forecast_vol = forecast_vol_arr[count]
        out = trade_best_option(cur_date, forecast_vol,
                                options_implied_vol_df, look_ahead=look_ahead,
                                long_only=False, direction=None,
                                atm_only=atm_only, trade_expiry=trade_expiry)
        pnl_series[count] = out['PnL']
        options_traded[count] = out['Trade_Type']
        option_type[count] = out['Option_Type']
        option_imp_vol[count] = out['Implied_Vol']
        if count % 100 == 0:
            print('\nProcessed', dates[count])
    end = time.time()
    print('Total Time taken is', end-start)
    model_df['PnL'] = pnl_series
    model_df['Cum_PnL'] = model_df['PnL'].fillna(0).cumsum()
    model_df['Options_Traded'] = options_traded
    model_df['Option_Type'] = option_type
    model_df['Option_Imp_Vol'] = option_imp_vol
    final_cum_pnl = model_df['Cum_PnL'].iloc[-1]
    print('The final Cumulative PnL is ${:.2f}'.format(final_cum_pnl))
    # Plot the cumulative PnL of the strategy
    plt.plot(model_df.index, model_df['Cum_PnL'])
    plt.xticks(rotation=90.)
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Cumulative PnL($)')
    plt.title(plot_title)
    plt.savefig('../Results/' + plot_title + '.jpg')
    
    return model_df
# %%
#Feature Engineering
def feature_normalization(df, col_names, train_date_end, scale_down=1,
                          percentile_flag=False):
    '''
    Implement z-score normalization using the mean and std-dev of the training
    data
    '''
    #df, col_names = regression_df.copy(), ['PUT_CALL_VOLUME_RATIO_CUR_DAY',
    #                                       'PX_VOLUME', 'Time_to_Expiry']
    for col in col_names:
        filter_train = df['Dates'] <= train_date_end
        train_data = df[col][filter_train]
        if percentile_flag:
            df[col] = df.apply(lambda x: percentileofscore(sorted(train_data),
                                                           x[col]),
                               axis='columns')
        else:
            mu = np.nanmean(train_data)
            std_dev = np.nanstd(train_data, ddof=1)
            df[col] = (df[col] - mu)/std_dev
    
    df[col_names] = df[col_names]/scale_down
    
    return df
# %%
###############################################################################
## Scrapers
###############################################################################        
def extract_words_pdf(filepath):
    '''
    Extracts the words from the pdf file
    '''
    import PyPDF2
    pdf_file = open(filepath, 'rb')
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    number_of_pages = read_pdf.getNumPages()
    words_list = []
    for n in range(number_of_pages):
        page = read_pdf.getPage(n)
        page_content = page.extractText()
        page_words_list = page_content.split('\n')
        page_words_list = [w.strip() for w in page_words_list if w.isupper()]
        words_list += page_words_list

    return words_list

def z_score(gtrend_row):
    '''
    Returns Z-Score
    '''
    gtrend_row = (gtrend_row-np.nanmean(gtrend_row))/np.nanstd(gtrend_row,
                                                               ddof=1)
    return gtrend_row

def google_trends(keyword_list=["Blockchain"],cat= 12,
                  time_frame ="2000-01-01 2017-12-31",
                  gprop = "",make_plot=False, geo='US'):
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
    pytrends.build_payload(kw_list=keyword_list,cat=cat, timeframe="2004-01-01 2007-12-31",
                           geo=geo, gprop='')
    df1=pytrends.interest_over_time()
    
    pytrends.build_payload(kw_list=keyword_list,cat=cat, timeframe="2008-01-01 2012-12-31",
                           geo=geo, gprop='')
    df2=pytrends.interest_over_time()
    
    pytrends.build_payload(kw_list=keyword_list,cat=cat, timeframe="2013-01-01 2017-12-31",
                           geo=geo, gprop='')
    df3=pytrends.interest_over_time()
    
    df = pd.concat([df1,df2,df3])
    df.index  = df.index.shift(-2,"1D")
    
    if make_plot == True:
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
            
     