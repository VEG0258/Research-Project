import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from scipy.io import savemat
from scipy.io import loadmat
import pylab
import h5py
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from copy import copy
import random
import sys
import pickle

def model(numstates, n_neurons, batch_size, i_train, t_train, i_test, t_test):
    #setting up the network
    input_tensor = Input(batch_shape=(batch_size,None,numstates))#,batch_size=batch_size)
    lstm_layer = LSTM(n_neurons, input_shape=(None,numstates), recurrent_activation='tanh'
                    ,activation='tanh', return_sequences=True, return_state=True, stateful=True)(input_tensor)
    dense_layer = keras.layers.TimeDistributed(Dense(numstates,activation='softmax'))(lstm_layer[0])

    #compiling the network
    model = Model(inputs=[input_tensor], outputs=[dense_layer])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['categorical_crossentropy', 'accuracy']
    )

    #training the network
    cbnan=keras.callbacks.TerminateOnNaN()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,mode='min',restore_best_weights=True)
    checkpoint = keras.callbacks.ModelCheckpoint(f'models/Katherine_lstm_lag200_seq500_N512', monitor="val_loss", verbose=0, save_best_only=True, save_weights_only=False, mode="min", save_freq="epoch", initial_value_threshold=None) 
    history = model.fit(i_train, t_train, epochs=5000, validation_data=(i_test,t_test),callbacks=[cbnan,checkpoint,callback],shuffle=False,batch_size=batch_size)

    return history, model