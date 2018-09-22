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
options_implied_vol_df = pd.read_csv("../../Data/Options_Implied_Vol.csv")
spx_data_df = pd.read_excel('../../Data/Data_Dump_BBG.xlsx',
                            sheet_name='SPX Index', skiprows=4)
price_history_df = pd.read_excel('../../Data/Data_Dump_BBG.xlsx',
                                 sheet_name='Price History', skiprows=3)
price_history_df.drop(index=[0,1], inplace=True)
price_history_df.rename(columns={price_history_df.columns[0]: 'Dates'},
                                 inplace=True)
price_history_df['Dates'] = pd.to_datetime(price_history_df['Dates'])
bbg_data_df = pd.merge(spx_data_df, price_history_df, on='Dates', how='outer')
options_implied_vol_df = options_implied_vol_data_clean(options_implied_vol_df)
options_implied_vol_df = combine_data(options_implied_vol_df, bbg_data_df)
#fridays_list = list(options_implied_vol_df.resample('W-Fri',
#                                                    on='date')['date'].last())
spx_data = spx_data_df[['Dates', 'PX_LAST']]
spx_data['PX_LAST'].fillna(method='ffill', inplace=True)
spx_data.rename(columns={'PX_LAST':'SPX'}, inplace=True)
spx_data["Dates"] =  pd.to_datetime(spx_data["Dates"])
spx_data["Returns"] = spx_data["SPX"].pct_change()
spx_data['Log_Returns'] = spx_data[['SPX']].apply(lambda x: np.log(x/x.shift(1)))
spx_data.dropna(inplace=True)
spx_data["Std Dev"] = spx_data["Returns"].rolling(5).std()
spx_data["Std Dev_EWMA"] = calculate_ewma_vol(spx_data["Returns"], 0.94, 5)
returns_series = spx_data["Returns"]
cum_mean_returns = returns_series.cumsum()/np.arange(1, len(returns_series)+1)
spx_data["Innovations_Squared"] = (returns_series - cum_mean_returns)**2
#%%
###############################################################################
## B. Variance Series Smoothing, and Baselining
###############################################################################
returns_series = returns_series.values
num_points = returns_series.size
train_idx = int(num_points*0.6)
cv_idx = int(num_points*0.85)
X_train = returns_series[:train_idx]
X_cv = returns_series[train_idx:cv_idx]
X_test = returns_series[cv_idx:]
dates = spx_data["Dates"][1:cv_idx]

# Scale values by a factor to ensure GARCH optimizer doesn't fail
scale_factor = 100
forecast_window = 1
fitted_result = fit_garch_model(ts=X_train*scale_factor)
#fitted_result = fit_garch_model(ts=np.append(X_train, X_cv)*scale_factor)
# Forecast forecast window ahead volatility on the cv set
init_resid = scale_factor*spx_data.iloc[train_idx]['Returns']-fitted_result.params.loc['mu']
init_vol = spx_data.iloc[train_idx]['Std Dev']*scale_factor
forecast_vol = forecast_garch(fitted_result,
                              spx_data[['Returns']][train_idx:cv_idx]*scale_factor,
                              init_resid, init_vol)/scale_factor
#Deprecated part
###############################################################################
#forecast_vol = fitted_result.forecast(horizon=forecast_window, start=train_idx,
#                                      align='target').variance
#forecast_vol.dropna(inplace=True)
##forecast_vol = np.sqrt(forecast_vol.mean(axis=1).values)/scale_factor
#colname = 'h.' + str(forecast_window)
#forecast_vol = np.sqrt(forecast_vol[colname].values)/scale_factor
###############################################################################

# Drop the 1st value since it's NaN
fitted_vol = fitted_result.conditional_volatility[1:]/scale_factor

# Plot Benchmark against Realized Vol for entire series
train_dates = dates[1:train_idx]
y_train = spx_data["Std Dev"][1:train_idx].values
plt.plot(train_dates, y_train, label = "Realized Volatilty")
plt.plot(train_dates, fitted_vol, label = "GARCH (benchmark)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=90.)
plt.title("Realized vs GARCH")
plt.savefig("../Results/Fitted_Realized_Vol.jpg")

# Plot forecast window ahead Benchmark volatility against Realized Vol
# for test set
forecast_dates = dates[(train_idx+forecast_window-1):]
forecast_vol = forecast_vol[1:]
y_cv_true = spx_data.loc[spx_data["Dates"].isin(forecast_dates),
                         "Std Dev"].values
plt.clf()
plt.plot(forecast_dates, y_cv_true, label = "Realized Volatilty")
plt.plot(forecast_dates, forecast_vol, label = "GARCH (benchmark)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=30.)
plt.title("Realized vs GARCH")
plt.savefig("../Results/Forecasted_Realized_Vol.jpg")

# Calculate Benchmark Values on the CV set (against volatility)
garch_cv_mse = np.mean((y_cv_true - forecast_vol)**2)
print('The Benchmark MSE on the cv is {:.2e}'.format(garch_cv_mse))

# Calculate Naive Values on the CV set (against volatility)
naive_cv_mse = np.mean((y_cv_true - np.nanmean(y_train))**2)
print('The Benchmark MSE on the cv is {:.2e}'.format(naive_cv_mse))

# Calculate the forecast df
forecast_df = pd.DataFrame(forecast_vol, forecast_dates, ['Forecast_Vol'])

# Calculate the realized df
realized_df = pd.DataFrame(y_cv_true, forecast_dates, ['Forecast_Vol'])
#%%
# Backtest the benchmark
benchmark_df = backtester(forecast_df, options_implied_vol_df,
                          'GARCH Back Test', look_ahead=7, atm_only=True)
benchmark_df.to_csv('GARCH Performance.csv')
#%%
# Backtest the realized vol
best_case_df = backtester(realized_df, options_implied_vol_df,
                          'Realized Back Test', look_ahead=1, atm_only=True,
                          trade_expiry=True)
best_case_df.to_csv('Realized Performance.csv')
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
   