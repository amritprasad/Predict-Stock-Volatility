"""
Neural Network Module
Author: Nathan Johnson
Date: 9/21/2018
"""
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Reshape, SimpleRNN, Lambda, Activation
from keras import layers, optimizers
import keras.backend as K
from keras.layers.advanced_activations import ELU
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
import random as rn


def run_jnn(lag_innov, innov, scale_factor,
            train_idx, cv_idx, batch_size=256, epochs=1000,
            plot_flag=False, jnn_isize=1, args_dict=None):
    '''
    Runs JNN(jnn_size)
    '''
    seed(42)
    set_random_seed(42)
    rn.seed(42)
    
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
    14) recurrent_reg_l1
    15) recurrent_reg_l2
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
    rec_reg_l1 = args_dict['recurrent_reg_l1']
    rec_reg_l2 = args_dict['recurrent_reg_l2']
    hidden_reg_b_l1_1 = args_dict['hidden_reg_b_l1_1']
    hidden_reg_b_l2_1 = args_dict['hidden_reg_b_l2_1']
    hidden_reg_b_l1_2 = args_dict['hidden_reg_b_l1_2']
    hidden_reg_b_l2_2 = args_dict['hidden_reg_b_l2_2']
    rnn_reg_b_l1 = args_dict['rnn_reg_b_l1']
    rnn_reg_b_l2 = args_dict['rnn_reg_b_l2']
    
    hidden_reg_1 = keras.regularizers.l1_l2(l1=hidden_reg_l1_1,
                                            l2=hidden_reg_l2_1)
    hidden_reg_2 = keras.regularizers.l1_l2(l1=hidden_reg_l1_2,
                                            l2=hidden_reg_l2_2)
    hidden_bias_reg_1 = keras.regularizers.l1_l2(l1=hidden_reg_b_l1_1,
                                                 l2=hidden_reg_b_l2_1)
    hidden_bias_reg_2 = keras.regularizers.l1_l2(l1=hidden_reg_b_l1_2,
                                                 l2=hidden_reg_b_l2_2)
    rnn_reg = keras.regularizers.l1_l2(l1=output_reg_l1, l2=output_reg_l2)
    recurrent_reg = keras.regularizers.l1_l2(l1=rec_reg_l1,
                                             l2=rec_reg_l2)
    rnn_bias_reg = keras.regularizers.l1_l2(l1=rnn_reg_b_l1, l2=rnn_reg_b_l2)
    
    #Define NN architecture
    h1_len, h2_len = 3, 2
    
    ilayer = layers.Input(shape=(input_len,))
    #layer_norm_i = layers.Lambda(layer_norm)(ilayer)
    batch_norm_i = layers.BatchNormalization(axis=1)(ilayer)
    hidden_1 = layers.Dense(h1_len, kernel_initializer=hidden_initializer,
                            #kernel_regularizer=hidden_reg_1,
                            activation='tanh',
                            #bias_regularizer=hidden_bias_reg_1
                           )(batch_norm_i)
    #elu_h_1 = layers.ELU(alpha=1.)(hidden_1)
    batch_norm_h_1 = layers.BatchNormalization(axis=1)(hidden_1)
    #layer_norm_1 = layers.Lambda(layer_norm)(hidden_1)
    #drop_1 = layers.Dropout(dropout_rate, seed=42)(batch_norm_h_1)
    #drop_1 = layers.Dropout(dropout_rate, seed=42)(elu_h_1)
    #resh_hidden_1 = layers.Reshape((1, 2))(layer_norm_1)
    #resh_hidden_1 = layers.Reshape((1, 5))(batch_norm_h_1)
    resh_hidden_1 = layers.Reshape((1, h1_len))(batch_norm_h_1)
    hidden_2 = layers.Dense(h2_len, kernel_initializer=hidden_initializer,
                            #kernel_regularizer=hidden_reg_2,
                            activation='tanh',
                            #bias_regularizer=hidden_bias_reg_2
                           )(resh_hidden_1)
    batch_norm_h_2 = layers.BatchNormalization(axis=1)(hidden_2)
    #elu_h_2 = layers.ELU(alpha=1.)(hidden_2)
    #layer_norm_2 = layers.Lambda(layer_norm)(hidden_2)
    #drop_2 = layers.Dropout(dropout_rate, seed=42)(batch_norm_h_2)
    #drop_2 = layers.Dropout(dropout_rate, seed=42)(hidden_2)
    #resh_hidden_2 = layers.Reshape((1, 2))(layer_norm_2)
    resh_hidden_2 = layers.Reshape((1, h2_len))(batch_norm_h_2)
    rnn = layers.SimpleRNN(1,
                           return_sequences=True,
                           activation=output_activation,
                           kernel_regularizer=rnn_reg,
                           kernel_initializer=rnn_initializer,
                           use_bias=True,
                           #recurrent_regularizer=recurrent_reg,
                           #bias_regularizer=rnn_bias_reg
                          )(resh_hidden_2)
    model = keras.models.Model(ilayer, rnn)

    #optimizer = optimizers.adam(optim_learning_rate,
                                 #epsilon=1e-5
    #                            )
    optimizer = optimizers.RMSprop(optim_learning_rate,
                                   #epsilon=1e-5
                                  )
#     optimizer = optimizers.SGD(optim_learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss
                  #loss=custom_error
                  )
    return model

def train_jnn(model, x_train, y_train, epochs=5, batch_size=100):
    model.fit(x_train, y_train, 
              epochs=epochs, batch_size=batch_size,
              verbose=0, shuffle=False)

def custom_error(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred), axis=0)

def get_model_gradients(model, inputs):
    weights = [tensor for tensor in model.trainable_weights]
    optimizer = model.optimizer
    
    gradients = optimizer.get_gradients(model.total_loss, weights)
    input_tensors = [model.inputs[0], #input data
                     model.targets[0], #target data
                     model.sample_weights[0], #set to [1] for full weighting
                     K.learning_phase() #test=0, train=1
                     ]
    get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    return get_gradients(inputs)

###############################################################################
#FFN
###############################################################################
def run_ffn(lag_innov, innov, scale_factor,
            train_idx, cv_idx, batch_size=256, epochs=1000, input_len=1,
            plot_flag=False, args_dict=None):
    '''
    Runs FFN(jnn_size)
    '''
    seed(42)
    set_random_seed(42)
    
    x_train, x_cv = lag_innov[:train_idx], lag_innov[train_idx:cv_idx]
    y_train, y_cv = innov[:train_idx], innov[train_idx:cv_idx]
    # Scale the variables to avoid the problem of vanishing gradients
    x_train, x_cv = x_train*scale_factor, x_cv*scale_factor
    y_train = y_train*scale_factor
    
    Y_train, Y_cv = prepare_tensors([y_train, y_cv])

    ffn = build_ffn(input_len, args_dict)
    train_ffn(ffn, x_train, Y_train, epochs=epochs, batch_size=batch_size)

    fit = ffn.predict(x_train).ravel()/scale_factor
    pred = ffn.predict(x_cv).ravel()/scale_factor
    mse = np.mean((y_cv - pred)**2)

    if plot_flag:
        plt.plot(np.sqrt(y_cv))
        plt.plot(np.sqrt(pred))
        plt.title("Volatility vs Predicted Volatility", fontsize=24)
        plt.legend(("Volatility", "Predicted"), fontsize=20)

    print('CV MSE is', mse)
    # return jnn.get_weights(), pred, mse
    return ffn, fit, pred, mse

def build_ffn(input_len, args_dict):
    '''
    Generates a keras Sequential model of an FFN    
    '''
    #input_len = 1
    learning_rate = args_dict['learning_rate']
    
    model = Sequential()
    model.add(Dense(1, input_shape=(input_len,),
                    kernel_initializer='he_normal'))
    model.add(ELU())
    #model.add(Dense(2, input_shape=(2,),
    #                kernel_initializer='he_normal'))
    model.add(Reshape((1, 1)))
    #model.add(layers.BatchNormalization())
    model.add(Dense(1, input_shape=(1,),
                    kernel_initializer='he_normal',
                    activation='linear'))
    #model.add(SimpleRNN(1, return_sequences=True,
    #                    kernel_initializer='he_normal', activation='linear'))
    
    #optimizer = optimizers.SGD(learning_rate)
    optimizer = optimizers.adam(learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error'
                  # loss=squared_error
                  )
    return model

def train_ffn(model, x_train, y_train, epochs=5, batch_size=100):
    model.fit(x_train, y_train, 
              epochs=epochs, batch_size=batch_size,
              verbose=1, shuffle=False)    
# %%

# %%