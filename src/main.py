# -*- coding: utf-8 -*-
'''
Expected to have sections which make function calls
'''

''' Imports '''

from helpers import *
from neural_network_module import *
from pandas.tseries.offsets import WeekOfMonth
import statsmodels.api as sm
import warnings

''' Options '''
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", 20)
warnings.filterwarnings('once')
#%%
###############################################################################
## A. Read data and clean
## Data could be Price series OHLCV, Risk Variance series
## 
###############################################################################
options_implied_vol_df = pd.read_csv("../../Data/Options_Implied_Vol_Weekly.csv")
spx_data_df = pd.read_excel('../../Data/Data_Dump_BBG.xlsx',
                            sheet_name='SPX Index', skiprows=4)
price_history_df = pd.read_excel('../../Data/Data_Dump_BBG.xlsx',
                                 sheet_name='Price History', skiprows=3)
trends_df = pd.read_csv('../../Data/trends_papers.csv')
news_predict_df = pd.read_csv('../../Data/predict_proba.csv',)
#Process Google Trends data
trends_df.rename(columns={'date':'Dates'}, inplace=True)
trends_df['Dates'] = pd.to_datetime(trends_df['Dates'])
trends_df.set_index('Dates', inplace=True)
cols_to_process = trends_df.columns
trends_df = google_trends_features(trends_df, cols_to_process, window=6)
#Process News Data
news_predict_df.drop(columns=news_predict_df.columns[0], inplace=True)
news_predict_df['Date'] = pd.to_datetime(news_predict_df['Date'])
news_predict_df.set_index('Date', inplace=True)
price_history_df.drop(index=[0,1], inplace=True)
price_history_df.rename(columns={price_history_df.columns[0]: 'Dates'},
                                 inplace=True)
price_history_df['Dates'] = pd.to_datetime(price_history_df['Dates'])
bbg_data_df = pd.merge(spx_data_df, price_history_df, on='Dates', how='outer')
options_implied_vol_df = options_implied_vol_data_clean(options_implied_vol_df,
                                                        weekly_flag=True)
options_implied_vol_df = combine_data(options_implied_vol_df, bbg_data_df)
#Construct features
wom = WeekOfMonth(week=3, weekday=4)
bbg_data_df['Time_to_Expiry'] = bbg_data_df['Dates'].apply(lambda x:
                                                          (wom.rollforward(x)
                                                           - x).days)
spx_data = bbg_data_df[['Dates', 'PX_LAST', 'Time_to_Expiry',
                        'OPEN_INT_TOTAL_PUT', 'OPEN_INT_TOTAL_CALL',
                        'PX_VOLUME', 'VIX Index']]
spx_data['PX_LAST'].fillna(method='ffill', inplace=True)
spx_data.rename(columns={'PX_LAST':'SPX'}, inplace=True)
spx_data["Dates"] =  pd.to_datetime(spx_data["Dates"])
spx_data["Returns"] = spx_data["SPX"].pct_change()
spx_data['Log_Returns'] = spx_data[['SPX']].apply(lambda x: np.log(x/x.shift(1)))
spx_data.dropna(inplace=True)
spx_data["Std Dev"] = spx_data["Returns"].rolling(5).std()
spx_data['Variance'] = spx_data['Std Dev']**2
returns_series = spx_data["Returns"]
cum_mean_returns = returns_series.cumsum()/np.arange(1, len(returns_series)+1)
spx_data['Innovations'] = returns_series - cum_mean_returns
spx_data["Innovations_Squared"] = spx_data['Innovations']**2
regression_df = spx_data.resample('W-Fri', on='Dates').last()
cols_to_sum = ['OPEN_INT_TOTAL_PUT', 'OPEN_INT_TOTAL_CALL', 'PX_VOLUME']
regression_df.drop(columns=cols_to_sum, inplace=True)
temp_df = spx_data[cols_to_sum + ['Dates']].resample('W-Fri', on='Dates').sum()
regression_df = regression_df.merge(temp_df, left_index=True,
                                    right_index=True, how='outer')
print('Checking how clean the regression_df is.\nFollowing is the no. of NAs-')
print(regression_df.isna().sum())
regression_df.dropna(inplace=True)
y = regression_df['Variance'].values[1:].copy()
X = regression_df[['Variance', 'Innovations_Squared']].values[:-1].copy()
X = sm.add_constant(X)
#%%
###############################################################################
## B. Variance Series Smoothing, and Baselining
###############################################################################
train_end_date = pd.to_datetime('2010-10-22')
cv_end_date = pd.to_datetime('2015-04-17')
test_end_date = pd.to_datetime('2017-12-29')
train_idx = sum(regression_df.index[1:] <= train_end_date)
cv_idx = sum(regression_df.index[1:] <= cv_end_date)
test_idx = sum(regression_df.index[1:] <= test_end_date)
X_train, y_train = X[:train_idx], y[:train_idx]
X_cv, y_cv = X[train_idx:cv_idx], y[train_idx:cv_idx]
X_test, y_test = X[cv_idx:], y[cv_idx:]
dates = regression_df['Dates'][1:]

###############################################################################
# B.1. Benchmark calculations
###############################################################################
garch_result = sm.OLS(y_train, X_train).fit()
garch_params = garch_result.params
#Forecast on the cv set using the fitted parameters
#Take square root to convert variances to vol
y_cv_benchmark = np.sqrt(X_cv @ garch_params)
y_train_benchmark = np.sqrt(X_train @ garch_params)

###############################################################################
# B.2. Naive calculations
###############################################################################
#Forecast using the naive model
y_cv_naive = np.mean(np.sqrt(y_train))

###############################################################################
# B.3. Neural Network calculations
###############################################################################
#Feature Engineering
#Combine regression_df with Google trends Data
regression_df = pd.merge(regression_df, trends_df, left_index=True,
                         right_index=True, how='outer')
#BackFill Google Trends
modes = trends_df.loc[trends_df.index.isin(dates[:train_idx])].mode()
modes = modes.to_dict('records')[0]
#regression_df.update(regression_df[trends_df.columns].fillna(modes))
regression_df.update(regression_df[trends_df.columns].fillna(0))
#Combine with news predictions
regression_df = regression_df.merge(news_predict_df, left_index=True,
                                    right_index=True, how='outer')
news_cols = ['avg_neg_proba', 'avg_pos_proba']
regression_df.update(regression_df[news_cols].fillna(0.5))
regression_df = regression_df[regression_df.index <= test_end_date]
#Continue Feature Engineering
regression_df['down_returns'] = [min(x, 0) for x in regression_df['Returns']]
regression_df['down_innov'] = [min(x, 0) for x in regression_df['Innovations']]
train_dates = dates[:train_idx]
train_date_end = train_dates[-1]
cols_to_normalize = ['PX_VOLUME', 'VIX_change', 'plunges_change',
                     'stocks_change', 'drop_change']
regression_df = feature_normalization(regression_df, cols_to_normalize,
                                      train_date_end, scale_down=100,
                                      percentile_flag=True)
std_dev = feature_normalization(regression_df, ['Std Dev'],
                                train_date_end, scale_down=100,
                                percentile_flag=True)
# %%
#Final Testing
X_train_cv, y_train_cv = X[:cv_idx], y[:cv_idx]
###############################################################################
# C.1. Benchmark calculations
###############################################################################
final_garch_result = sm.OLS(y_train_cv, X_train_cv).fit()
final_garch_params = final_garch_result.params
#Forecast on the cv set using the fitted parameters
#Take square root to convert variances to vol
y_train_cv_benchmark = np.sqrt(X_train_cv @ final_garch_params)
y_test_benchmark = np.sqrt(X_test @ final_garch_params)

###############################################################################
# C.2. Naive calculations
###############################################################################
#Forecast using the naive model
y_test_naive = np.mean(np.sqrt(y_train_cv))
# %%
###############################################################################
# C.3. NN calculations (No Need to run if the model's already trained)
###############################################################################
scale_factor = 1E2
lag_innov = np.sqrt(X[:, 1])
lag_innov = np.column_stack((lag_innov,
                             regression_df['PX_VOLUME'].values[:-1]))
lag_innov = np.column_stack((lag_innov,
                             regression_df['stocks_change'].values[:-1]))
num_nn_inputs = lag_innov.shape[1] if lag_innov.ndim > 1 else 1
innov = np.sqrt(y)
num_epochs, batch_size, learning_rate = 30000, cv_idx, 0.008
args_dict = {
             'hidden_initializer': keras.initializers.he_normal(seed=42),
             #'hidden_initializer': keras.initializers.he_normal(seed=42),
             'dropout_rate': 0.2,
             'rnn_initializer': keras.initializers.he_normal(seed=42),
             #'rnn_initializer': 'he_normal',
             'optim_learning_rate': learning_rate,
             'loss': 'mean_squared_error',
             #'loss': custom_error,
             'hidden_reg_l1_1': 0.,
             'hidden_reg_l2_1': 0.,
             'hidden_reg_l1_2': 0.,
             'hidden_reg_l2_2': 0.,
             'output_reg_l1': 0.,
             'output_reg_l2': 0.,
             'hidden_activation': ELU(alpha=1.),
             #'hidden_activation': PReLU(),
             'output_activation': 'linear',
             'recurrent_reg_l1': 0.,
             'recurrent_reg_l2': 0.,
             'hidden_reg_b_l1_1': 0.,
             'hidden_reg_b_l2_1': 0.,
             'hidden_reg_b_l1_2': 0.,
             'hidden_reg_b_l2_2': 0.,
             'rnn_reg_b_l1': 0.,
             'rnn_reg_b_l2': 0.
            }
from neural_network_module import *
jnn_trained, nn_fit_vol, nn_forecast_vol, _ = run_jnn(lag_innov, innov,
                                                      scale_factor,
                                                      cv_idx, test_idx,
                                                      batch_size=batch_size,
                                                      epochs=num_epochs,
                                                      plot_flag=False,
                                                      jnn_isize=num_nn_inputs,
                                                      args_dict=args_dict)
#Plot Loss vs Epochs
plt.clf()
plt.rcParams["figure.figsize"] = (10, 8)
jnn_history = jnn_trained.history
plot_epochs, plot_loss = [], []
for idx, loss in enumerate(jnn_history.history['loss']):
#    if loss < 1e-4:
    plot_loss.append(loss*1E5)
    plot_epochs.append(jnn_history.epoch[idx])
plt.plot(plot_epochs[1000:], plot_loss[1000:], label="Loss vs Epochs")
#plt.plot(plot_epochs, plot_loss, label="Loss vs Epochs")
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Loss')
title_str = 'Epochs- '+str(num_epochs)+', Batch- '+str(batch_size)+', lr- '+str(learning_rate)
plt.title(title_str)
#plt.savefig('')
jnn_weights = jnn_trained.get_weights()
jnn_inputs = [np.reshape(lag_innov[:train_idx], (train_idx, num_nn_inputs)),
              np.reshape(innov[:train_idx], (train_idx, 1, 1)), [1], 0]
get_model_gradients(jnn_trained, jnn_inputs)
# %%
try:
    jnn_trained
except NameError:
    filename = "model.h5"
    jnn_trained = load_model(filename)
    scale_factor = 1E2
    input_ = np.sqrt(X[:, 1])    
    input_ = np.column_stack((input_,
                              regression_df['PX_VOLUME'].values[:-1]))
    input_ = np.column_stack((input_,
                              regression_df['stocks_change'].values[:-1]))
    #input_cv = np.column_stack((input_cv, downside_returns))
    input_ = input_*scale_factor
    input_train_cv = input_[:cv_idx]
    nn_fit_vol = jnn_trained.predict(input_train_cv).ravel()/scale_factor
    input_test = input_[cv_idx:]    
    nn_forecast_vol = jnn_trained.predict(input_test).ravel()/scale_factor
else:
    print('NN was defined')
# Plot Benchmark against Realized Vol for trained series
train_cv_dates = dates[:cv_idx]
y_train_cv_true = regression_df.loc[regression_df["Dates"].isin(train_cv_dates),
                                    "Std Dev"].values
plt.rcParams["figure.figsize"] = (10, 8)
plt.plot(train_cv_dates, y_train_cv_true, label="Realized Volatilty")
plt.plot(train_cv_dates, y_train_cv_benchmark, label="GARCH (benchmark)")
plt.plot(train_cv_dates, nn_fit_vol, label="Latest State of the Art",
         marker='_', color='moccasin')
plt.legend()
plt.grid(True)
plt.xticks(rotation=30.)
plt.title("Realized vs GARCH vs State of the Art (Fitted)")
plt.savefig("../Results/Fitted_Comparison_Vol (Train + CV).jpg")

# Plot forecast window ahead Benchmark and NN volatility against Realized Vol
# for cv set
forecast_test_dates = dates[cv_idx:]
y_test_true = regression_df.loc[regression_df["Dates"].isin(forecast_test_dates),
                                "Std Dev"].values
plt.clf()
plt.rcParams["figure.figsize"] = (10, 8)
plt.plot(forecast_test_dates, y_test_true, label = "Realized Volatilty")
plt.plot(forecast_test_dates, y_test_benchmark, label = "GARCH (benchmark)",
         marker='.')
plt.plot(forecast_test_dates, nn_forecast_vol, label = "Latest State of the Art",
         marker='_', color='moccasin')
plt.legend()
plt.grid(True)
plt.xticks(rotation=30.)
plt.title("Realized vs GARCH vs State of the Art")
plt.savefig("../Results/Forecast_Comparison_Vol (Test).jpg")

# Calculate Benchmark Values on the CV set (against volatility)
garch_test_mse = np.mean((y_test_benchmark - y_test_true)**2)
print('The Benchmark MSE on the test is {:.2e}'.format(garch_test_mse))

# Calculate NN Values on the CV set (against volatility)
nn_test_mse = np.mean((y_test_true - nn_forecast_vol)**2)
print('The NN MSE on the test is {:.2e}'.format(nn_test_mse))

# Calculate Naive Values on the CV set (against volatility)
naive_test_mse = np.mean((y_test_true - y_test_naive)**2)
print('The Naive MSE on the test is {:.2e}'.format(naive_test_mse))

# Calculate the forecast df
forecast_test_df = pd.DataFrame(y_test_benchmark, forecast_test_dates,
                                ['Forecast_Vol'])

# Calculate the realized df
realized_test_df = pd.DataFrame(y_test_true, forecast_test_dates, ['Forecast_Vol'])

# Calculate the NN forecast df
nn_forecast_test_df = pd.DataFrame(nn_forecast_vol, forecast_test_dates,
                                   ['Forecast_Vol'])

# Calculate the naive df
naive_forecast_test_df = pd.DataFrame(y_test_naive, forecast_test_dates,
                                      ['Forecast_Vol'])
#%%
# Backtest the benchmark
benchmark_df, _ = backtester(forecast_test_df, options_implied_vol_df,
                             'GARCH Back Test_Test Set', look_ahead=7,
                             atm_only=True, trade_expiry=True)
benchmark_df.to_csv('GARCH Performance_Test Set.csv')
#%%
# Backtest the realized vol
best_case_df, _ = backtester(realized_test_df, options_implied_vol_df,
                             'Realized Back Test_Test Set', look_ahead=7,
                             atm_only=True, trade_expiry=True)
best_case_df.to_csv('Realized Performance_Test Set.csv')
#%%
# Backtest the Neural Net
nn_df, _ = backtester(nn_forecast_test_df, options_implied_vol_df,
                      'Neural Net Back Test_Test Set', look_ahead=7,
                      atm_only=True, trade_expiry=True)
nn_df.to_csv('NN Performance_Test Set.csv')
# %%
# Backtest the realized vol
naive_case_df, _ = backtester(naive_forecast_test_df, options_implied_vol_df,
                              'Realized Back Test_Test Set', look_ahead=7,
                              atm_only=True, trade_expiry=True)
best_case_df.to_csv('Naive Performance_Test Set.csv')
# %%
#Plot all 3 together
plt.plot(naive_case_df['Cum_PnL'])
plt.plot(benchmark_df['Cum_PnL'])
plt.plot(nn_df['Cum_PnL'])
plt.legend(['Naive', 'GARCH', 'NN'])
plt.xlabel('Dates')
plt.ylabel('Cumulative P&L ($)')
plt.title('Comparison of Trading Strategies')
plt.grid(True)
plt.savefig('../Results/Compare_Back_Test_Test Set.jpg')
# %%
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
predict_proba = pd.read_csv("../../Data/predict_proba.csv")
predict_proba['Date'] = pd.to_datetime(predict_proba['Date'])
predict_proba_train = predict_proba[predict_proba['Date'].isin(train_dates)]
predict_proba_train = predict_proba_train['avg_neg_proba'].values
predict_proba_cv = predict_proba[predict_proba['Date'].isin(forecast_dates)]
predict_proba_cv = predict_proba_cv['avg_neg_proba'].values
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

filenames = ['LM_Negative.pdf', 'LM_Positive.pdf', 'LM_Uncertainty.pdf']
directory = 'D:/MFE/Semester Courses/Fall/230T/Project/Sentiment/'
neg_words_list = extract_words_pdf(os.path.join(directory, filenames[0]))
pos_words_list = extract_words_pdf(os.path.join(directory, filenames[1]))
uncert_words_list = extract_words_pdf(os.path.join(directory, filenames[2]))

positive_words = pos_words_list + ["bulls"]
negative_words = neg_words_list + ["bears"]
negative_words = [x.split(' ')[-1] for x in negative_words]
negative_words = ['LOSS', 'LOSSES', 'CLAIMS', 'IMPAIRMENT', 'AGAINST',
                  'ADVERSE', 'RESTATED', 'ADVERSELY', 'RESTRUCTURING',
                  'LITIGATION', 'DISCONTINUED', 'TERMINATION', 'DECLINE',
                  'CLOSING', 'FAILURE', 'UNABLE', 'DAMAGES', 'DOUBTFUL',
                  'LIMITATIONS', 'FORCE', 'VOLATILITY', 'CRITICAL',
                  'TERMINATED', 'IMPAIRED', 'COMPLAINT', 'DEFAULT',
                  'NEGATIVE', 'DEFENDANTS', 'PLAINTIFFS', 'DIFFICULT']
uncertain_words = uncert_words_list
scrape_these_words(key_words =positive_words[:5] ,path = "../../Data",
                       file_name = "positive_words_2000_v2.csv")

scrape_these_words(key_words =negative_words[:5] ,path = "../../Data",
                       file_name = "negative_words_2000_v2.csv")

scrape_these_words(key_words =negative_words ,path = "../../Data",
                       file_name = "uncertain_words_2000_v2.csv")
