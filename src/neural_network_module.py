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
from numpy.random import seed
from tensorflow import set_random_seed


def run_jnn(lag_innov, innov, stddev_window,
            train_idx, cv_idx, batch_size=256, epochs=1000,
            plot_flag=False, jnn_size=(1, 1, 1)):
    '''
    Runs JNN(jnn_size)
    '''
    seed(42)
    set_random_seed(42)
    i, h, o = jnn_size
    
    x_train, x_cv = lag_innov[:train_idx], lag_innov[train_idx:cv_idx]
    y_train, y_cv = innov[:train_idx], innov[train_idx:cv_idx]

    Y_train, Y_cv = prepare_tensors([y_train, y_cv])

    jnn = build_jnn(i, h, o)
    train_jnn(jnn, x_train, Y_train, epochs=epochs, 
              # batch_size=len(x_train),
              batch_size=batch_size)

    fit = jnn.predict(x_train).ravel()
    pred = jnn.predict(x_cv).ravel()
    mse = np.mean((y_cv - pred)**2)

    if plot_flag:
        plt.plot(np.sqrt(y_cv))
        plt.plot(np.sqrt(pred))
        plt.title("Volatility vs Predicted Volatility", fontsize=24)
        plt.legend(("Volatility", "Predicted"), fontsize=20)

    print('CV MSE is', mse)
    # return jnn.get_weights(), pred, mse
    return jnn, fit, pred, mse

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
    resh = layers.Reshape((1, hidden_node_num))(drop)
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
# %%

# %%