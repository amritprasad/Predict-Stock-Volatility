"""
Neural Network Module
Author: Nathan Johnson
Date: 9/21/2018
"""
import keras
from keras import layers
from keras import optimizers
import keras.backend as K
from keras.layers.advanced_activations import ELU
import numpy as np
import matplotlib.pyplot as plt


def run_example(lag_innov, innov, stddev_window,
                train_idx, cv_idx, batch_size=256, epochs=1000):
    '''
    Runs JNN(1,1,1)
    '''
    x_train, x_cv = lag_innov[(stddev_window-1):train_idx], lag_innov[train_idx:cv_idx]
    y_train, y_cv = innov[(stddev_window-1):train_idx], innov[train_idx:cv_idx]
    
    Y_train, Y_cv = prepare_tensors([y_train, y_cv])
    
    jnn = build_jnn(1, 1, 1)
    train_jnn(jnn, x_train, Y_train, epochs=epochs, 
              #batch_size=len(x_train),
              batch_size=batch_size)

    pred = jnn.predict(x_cv).ravel()
    mse = np.mean((y_cv - pred)**2)
    
    plt.plot(np.sqrt(y_cv))
    plt.plot(np.sqrt(pred))
    plt.title("Volatility vs Predicted Volatility", fontsize=24)
    plt.legend(("Volatility", "Predicted"), fontsize=20)
    print('CV MSE is', mse)
    return jnn.get_weights(), pred, mse

def prepare_tensors(array_list):
    return [np.reshape(array, (len(array), 1, 1)) for array in array_list]

def build_jnn(input_len, hidden_node_num, output_len):
    #generates a keras Sequential model of a JNN(p, q, t)
    #input_len, hidden_node_num, output_len = 1, 1, 1
    ilayer = layers.Input(shape=(input_len,))
    hidden = layers.Dense(hidden_node_num, 
              kernel_initializer='he_normal',
              #activation='sigmoid'
              activation=ELU()
              )(ilayer)
    drop = layers.Dropout(0.2, seed=42)(hidden)
    resh = layers.Reshape((input_len, hidden_node_num))(drop)    
    rnn = layers.SimpleRNN(output_len,
                           return_sequences=True,
                           activation='linear',
#                           activation=ELU(),
                           kernel_initializer='he_normal')(resh)
    model = keras.models.Model(ilayer, rnn)

    #optimizer = optimizers.adam(lr = 0.2)
    optimizer = optimizers.adam()
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error'
                  #loss=squared_error
                  )
    return model

def train_jnn(model, x_train, y_train, epochs=5, batch_size=100):
    model.fit(x_train, y_train, 
              epochs=epochs, batch_size=batch_size,
              verbose=1)

def squared_error(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred), axis=0)
#%%
stddev_window = 5
lag_innov = spx_data["Returns"].rolling(stddev_window).std().values
#7.736968823784901e-06
#lag_innov = spx_data['Log_Returns']
#lag_innov = spx_data["Returns"]
#lag_innov = (lag_innov - lag_innov.cumsum()/np.arange(1, len(lag_innov)+1))**2
#lag_innov = lag_innov.values
#lag_innov = np.stack((lag_innov.values, spx_data['Std Dev'].values)).T
#7.652951361697178e-08
#lag_innov = (lag_innov - lag_innov.rolling(stddev_window).mean())**2
#5.946558428284477e-08
#lag_innov = lag_innov*1E6
#lag_innov = lag_innov[~np.isnan(lag_innov)]
#lag_innov = calculate_ewma_vol(spx_data["Returns"], 0.94, 5)
#4.36e-05
#lag_innov.fillna(0, inplace=True)
#innov = spx_data["Log_Returns"].rolling(stddev_window).std().values[1:]
innov = spx_data["Returns"].rolling(stddev_window).std().values[1:]
#innov = lag_innov[1:]
lag_innov = lag_innov[:-1]
jnn_weights, pred, mse = run_example(lag_innov, innov, stddev_window,
                                     train_idx, cv_idx, 256, 1000)
#%%
#Forecast using NN innovation values
nn_forecast_vol = forecast_nn(fitted_result, init_resid, init_vol,
                              nn_innovations=pred*(scale_factor**2))
nn_forecast_vol = nn_forecast_vol[1:]/scale_factor
# Calculate NN Values on the CV set (against volatility)
nn_cv_mse = np.mean((y_cv_true - nn_forecast_vol)**2)
print('The Benchmark MSE on the cv is {:.2e}'.format(nn_cv_mse))
#%%