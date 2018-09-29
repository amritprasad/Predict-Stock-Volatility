"""
Neural Network Module
Author: Nathan Johnson
Date: 9/21/2018
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, SimpleRNN, Lambda, Activation
from keras import layers
from keras import optimizers
import keras.backend as K
from keras.layers.advanced_activations import ELU
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf


def run_jnn(lag_innov, innov, scale_factor,
            train_idx, cv_idx, batch_size=256, epochs=1000,
            plot_flag=False, jnn_isize=1, args_dict=None):
    '''
    Runs JNN(jnn_size)
    '''
    seed(42)
    set_random_seed(42)
    
    x_train, x_cv = lag_innov[:train_idx], lag_innov[train_idx:cv_idx]
    y_train, y_cv = innov[:train_idx], innov[train_idx:cv_idx]
    # Scale the variables to avoid the problem of vanishing gradients
    x_train, x_cv = x_train*scale_factor, x_cv*scale_factor
    y_train = y_train*scale_factor
    
    Y_train, Y_cv = prepare_tensors([y_train, y_cv])

    jnn = build_jnn(jnn_isize, args_dict)
    train_jnn(jnn, x_train, Y_train, epochs=epochs, 
              # batch_size=len(x_train),
              batch_size=batch_size)

    fit = jnn.predict(x_train).ravel()/scale_factor
    pred = jnn.predict(x_cv).ravel()/scale_factor
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

def build_jnn_deprecated(input_len, hidden_node_num, output_len):
    #generates a keras Sequential model of a JNN(p, q, t)
    #input_len, hidden_node_num, output_len = 1, 1, 1
    def layer_norm(x):
        return tf.contrib.layers.layer_norm(x)
    
    model = Sequential()
    model.add(Dense(hidden_node_num, 
                    input_shape=(input_len,),
                    kernel_initializer='he_normal'))
    model.add(Activation(activation=ELU()))
    model.add(Lambda(layer_norm))
    model.add(Dropout(0.2, seed=42))
    model.add(Reshape((1, hidden_node_num)))
    model.add(SimpleRNN(output_len, 
                        return_sequences=True,
                        kernel_initializer='he_normal'))
    model.add(Activation(activation='linear'))

    #optimizer = optimizers.adam(lr = 0.2)
    optimizer = optimizers.adam()
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error'
                  # loss=squared_error
                  )
    return model

def build_jnn(input_len, args_dict):    
    '''
    Create the structure for the JNN. args_dict-
    1) hidden_initializer
    2) dropout_rate
    3) rnn_initializer
    4) optim_learning_rate
    5) loss
    6) hidden_reg_l1_1
    7) hidden_reg_l2_1
    8) hidden_reg_l1_2
    9) hidden_reg_l2_2
    10) output_reg_l1
    11) output_reg_l2
    12) hidden_activation
    13) output_activation
    '''
    #generates a keras Sequential model of a JNN(p, q, t)
    def layer_norm(x): #used in lambda layer for layer normalization
        return tf.contrib.layers.layer_norm(x)
    #input_len, hidden_node_num, output_len = 1, 1, 1
    # Extract the args
    hidden_initializer = args_dict['hidden_initializer']
    dropout_rate = args_dict['dropout_rate']
    rnn_initializer = args_dict['rnn_initializer']
    optim_learning_rate = args_dict['optim_learning_rate']
    loss = args_dict['loss']
    hidden_reg_l1_1 = args_dict['hidden_reg_l1_1']
    hidden_reg_l2_1= args_dict['hidden_reg_l2_2']
    hidden_reg_l1_2 = args_dict['hidden_reg_l1_1']
    hidden_reg_l2_2= args_dict['hidden_reg_l2_2']
    output_reg_l1 = args_dict['output_reg_l1']
    output_reg_l2 = args_dict['output_reg_l2']
    hidden_activation = args_dict['hidden_activation']
    output_activation = args_dict['output_activation']
    
    hidden_reg_1 = keras.regularizers.l1_l2(l1=hidden_reg_l1_1,
                                            l2=hidden_reg_l2_1)
    hidden_reg_2 = keras.regularizers.l1_l2(l1=hidden_reg_l1_2,
                                            l2=hidden_reg_l2_2)
    rnn_reg = keras.regularizers.l1_l2(l1=output_reg_l1, l2=output_reg_l2)
    ilayer = layers.Input(shape=(input_len,))
    #batch_norm_i = layers.BatchNormalization(mode=0, axis=1)(ilayer)
    hidden_1 = layers.Dense(2, kernel_initializer=hidden_initializer,
                            kernel_regularizer=hidden_reg_1,
                            activation=hidden_activation
                           )(ilayer)
    #layer_norm_1 = layers.Lambda(layer_norm)(hidden_1)
    #drop_1 = layers.Dropout(dropout_rate, seed=42)(layer_norm_1)
    #drop_1 = layers.Dropout(dropout_rate, seed=42)(hidden_1)
    #resh_hidden_1 = layers.Reshape((1, 2))(layer_norm_1)
    resh_hidden_1 = layers.Reshape((1, 2))(hidden_1)
    hidden_2 = layers.Dense(2, kernel_initializer=hidden_initializer,
                            kernel_regularizer=hidden_reg_2,
                            activation=hidden_activation)(resh_hidden_1)
    #layer_norm_2 = layers.Lambda(layer_norm)(hidden_2)
    #drop_2 = layers.Dropout(dropout_rate, seed=42)(layer_norm_2)
    #drop_2 = layers.Dropout(dropout_rate, seed=42)(hidden_2)
    resh_hidden_2 = layers.Reshape((1, 2))(hidden_2)
    rnn = layers.SimpleRNN(1,
                           return_sequences=True,
                           activation=output_activation,
                           kernel_regularizer=rnn_reg,
                           kernel_initializer=rnn_initializer,
                           use_bias=True)(resh_hidden_2)
    model = keras.models.Model(ilayer, rnn)

    optimizer = optimizers.adam(optim_learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss
                  #loss=custom_error
                  )
    return model

def train_jnn(model, x_train, y_train, epochs=5, batch_size=100):
    model.fit(x_train, y_train, 
              epochs=epochs, batch_size=batch_size,
              verbose=1, shuffle=False)

def custom_error(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred), axis=0)
# %%

# %%