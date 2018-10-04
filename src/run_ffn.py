#y = regression_df['Variance'].values[1:].copy()
#X = regression_df[['Variance', 'Innovations_Squared']].values[:-1].copy()
#X = sm.add_constant(X)
scale_factor = 1E2
#lag_innov = regression_df['Innovations_Squared'].values[:-1]
lag_innov = np.append(nn_fit_vol, nn_forecast_vol).copy()
lag_innov = np.column_stack((lag_innov,
                             regression_df['PX_VOLUME'].values[:cv_idx]))
#innov = regression_df['Innovations_Squared'].values[1:]
innov = regression_df['Std Dev'].values[1:]
num_nn_inputs = lag_innov.shape[1] if lag_innov.ndim > 1 else 1
num_epochs, batch_size, learning_rate = 10000, train_idx, 1e-4
ffn_args_dict = {
                 'learning_rate': learning_rate
                }
#del model
from neural_network_module import *
ffn_trained, nn_fit_innov, nn_forecast_innov, _ = run_ffn(lag_innov, innov,
                                                          scale_factor,
                                                          train_idx, cv_idx,
                                                          batch_size=batch_size,
                                                          epochs=num_epochs,
                                                          input_len=num_nn_inputs,
                                                          plot_flag=False,
                                                          args_dict=ffn_args_dict)
nn_innov_all = ffn_trained.predict(lag_innov*scale_factor).ravel()/scale_factor
plt.clf()
plt.rcParams["figure.figsize"] = (10, 8)
ffn_history = ffn_trained.history
plot_epochs, plot_loss = [], []
for idx, loss in enumerate(ffn_history.history['loss']):
    #if loss < 1e-4*scale_factor:
    plot_loss.append(loss*1E5/scale_factor)
    plot_epochs.append(ffn_history.epoch[idx])
plt.plot(plot_epochs[200:], plot_loss[200:], label="Loss vs Epochs")
#plt.plot(plot_epochs, plot_loss, label="Loss vs Epochs")
plt.grid(True)
plt.xlabel('Epochs')
plt.ylabel('Loss')
title_str = 'Epochs- '+str(num_epochs)+', Batch- '+str(batch_size)+', lr- '+str(learning_rate)
plt.title(title_str)
ffn_trained.get_weights()
ffn_inputs = [np.reshape(lag_innov[:train_idx], (train_idx, num_nn_inputs)),
              np.reshape(innov[:train_idx], (train_idx, 1, 1)), [1], 0]
get_model_gradients(ffn_trained, ffn_inputs)
# %%
#Evaluate training fit
train_dates = dates[:train_idx]
y_train_true = regression_df.loc[regression_df["Dates"].isin(train_dates),
                                 "Innovations_Squared"].values
#plt.rcParams["figure.figsize"] = (15, 10)
plt.plot(train_dates, y_train_true, label="Innovations")
plt.plot(train_dates, nn_fit_innov, label="FFN",
         marker='_', color='moccasin')
plt.legend()
plt.grid(True)
plt.xticks(rotation=30.)
plt.title("FFN Fit")
# %%
#Evaluate forecast
forecast_dates = dates[train_idx:cv_idx]
y_cv_true = regression_df.loc[regression_df["Dates"].isin(forecast_dates),
                              "Innovations_Squared"].values
plt.clf()
#plt.rcParams["figure.figsize"] = (15, 10)
plt.plot(forecast_dates, y_cv_true, label = "Innovations")
plt.plot(forecast_dates, nn_forecast_innov, label = "FFN",
         marker='_', color='moccasin')
plt.legend()
plt.grid(True)
plt.xticks(rotation=30.)
plt.title("Realized vs FFN")
# %%
