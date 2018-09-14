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
spx_data = pd.read_csv("../data/spx_data.csv")
spx_data["Dates"] =  pd.to_datetime(spx_data["Dates"])
spx_data["Returns"] = spx_data["SPX"].pct_change()
spx_data["Std Dev"] = spx_data["Returns"].rolling(5).std()

#%%
###############################################################################
## B. Variance Series Smoothing, and Baselining
###############################################################################

fitted_result = fit_garch_model(ts= spx_data["Returns"] .dropna())
fitted_vol = fitted_result.conditional_volatility
spx_data["Fitted Vol"] = np.nan
spx_data["Fitted Vol"][1:] = fitted_vol
plt.plot(spx_data["Dates"],spx_data["Std Dev"],label = "Realized Volatilty")
plt.plot(spx_data["Dates"],spx_data["Fitted Vol"],
         label = "GARCH (benchmark)")
plt.legend()
plt.grid()
plt.title("Realized vs GARCH")
plt.savefig("./Results/Fitted_Realized_Vol.jpg")
spx_data.to_csv("./Results/SPX_data_with_fitted_vol.csv")
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

###############################################################################
## Z. APPENDIX
## I. Trends Scraping
###############################################################################
positive_words = ["gainer","whistleblower","speedy","dubious","scraps",
                  "acknowledge","delisted","downs","boding","disappeared",
                  "botched","kongs","surely","resurgent","eos","hindered",
                  "leapt","grapple","heated","forthcoming","standpoint",
                  "exacerbated","steer","toptier","braking","jackets",
                  "featured","overcrowded","saddled","haul"
                  ]

negative_words = ["dating","birthrate","reacting","lofty","accelerators",
                  "falsified","bust","averaging","pages","championed",
                  "folded","trillions","santa","fourfold","wellknown",
                  "perfect","defaults","bottleneck","cloudy",
                  "strains","kicks","doubted","halving","retailing","abandon",
                  "depressing","specifications","businessmen","diluting"
                  ]

scrape_these_words(key_words =positive_words ,path = "../data",
                       file_name = "positive_words_2000.csv")

scrape_these_words(key_words =negative_words ,path = "../data",
                       file_name = "negative_words_2000.csv")
   