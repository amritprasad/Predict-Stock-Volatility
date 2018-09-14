# -*- coding: utf-8 -*-
'''
Expected to have sections which make function calls
'''

''' Imports '''

from helpers import *

''' Options '''
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", 20)
#%%
###############################################################################
## A. Read data and clean
## Data could be Price series OHLCV, Risk Variance series
## 
###############################################################################
options_implied_vol_df = pd.read_csv("../Data/Options_Implied_Vol.csv")
spx_data_df = pd.read_excel('../Data/Data_Dump_BBG.xlsx',
                            sheet_name='SPX Index', skiprows=4)
options_implied_vol_df = options_implied_vol_data_clean(options_implied_vol_df)
options_implied_vol_df = combine_data(options_implied_vol_df, spx_data_df)
fridays_list = list(options_implied_vol_df.resample('W-Fri',
                                                    on='date')['date'].last())
#%%
###############################################################################
## B. Variance Series Smoothing, and Baselining
###############################################################################
smoothed_sp = pd.read_csv("../../data/snp_with_rv_bv.csv")
## Column names: ['Dates', 'Price', 'weekday', 'Return', 'Return(-1)', 
## 'Return(-2)','Return(-3)', 'Return(-4)', 'mu', 'RV', 'BV', 'k']
fitted_result = fit_garch_model(ts= smoothed_sp["Return"].dropna())

#%%
###############################################################################
## C. Feature Creation
##
###############################################################################


###############################################
## C. Feature Creation
## a. Technical Data Creation
###############################################


###############################################
## C. Feature Creation
## b. NLP Data Feature Creation
###############################################


###############################################
## C. Feature Creation
## c. NLP Data Feature Creation
###############################################