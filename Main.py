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

#import subclasses
import Methods 
import DataPreparing
import Model

numFlies=59
numstates=117
sequence_length=500
n_neurons=512
batch_size=59
lag=200

name_mat='behaviorTimeSeries.mat'
states='reduced_states'

### Traning ###############################################################################################################################################################

#data preparing 
#paramters: numFlies, numstates, sequence_length, n_neurons, batch_size, lag, name_mat, states
#returns: i_train, t_train, i_test, t_test
i_train, t_train, i_test, t_test = DataPreparing(numFlies, numstates, sequence_length, n_neurons, batch_size, lag, name_mat, states)

#model
#parameters: numstates, n_neurons, batch_size, i_train, t_train, i_test, t_test
#returns: history, model
history, model = Model(numstates, n_neurons, batch_size, i_train, t_train, i_test, t_test)

#save history
filename = f'history/Katherine_RNN_lag200.pkl'
with open(filename, 'wb') as f:
    pickle.dump(history.history, f)

# #access history
# with open(f'history/model_training_history_lag{i}.pkl', 'rb') as f:
#     loaded_history = pickle.load(f)

#loss plots
model.save(f'models/Katherine_lstm_lag500_seq500_N512')
plt.plot(history.history['loss'][:])
plt.plot(history.history['val_loss'][:])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig(f'loss/Katherine_loss_lag500')

#training done
print("Training is done")

### Prediction ####################################################################################################################################