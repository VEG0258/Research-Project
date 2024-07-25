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

def DataPreparing(numFlies, numstates, sequence_length, n_neurons, batch_size, lag, name_mat, states):
    #this loops sets the size of the numpy array that will contain the data
    minlength=500000
    mat_file = loadmat(name_mat)
    for i,x in enumerate(mat_file[states]):
        if np.size(x[0])<minlength:
            minlength=np.size(x[0])

    #putting the data into a numpy array
    dt = np.zeros((numFlies,minlength))
    for i,x in enumerate(mat_file[states]):
        dt[i,:]=x[0][0:minlength,0]-1
        indices = np.where(dt[i,:]==255)[0]
        indices2 = np.where(np.diff(indices)>1)[0]
        for idx in indices:
            if idx==0:
                dt[i,0]=np.random.randint(0,high=numstates)
            else:
                dt[i,idx]=copy(dt[i,idx-1])

    #batching the data
    v=.8
    inp,targ=Methods.get_batches(dt,batch_size,sequence_length,lag=lag)

    #breaking the data into training and test sets
    input_train=inp[:int(v*inp.shape[0]),:]
    input_test=inp[int(v*inp.shape[0]):,:]
    target_train=targ[:int(v*inp.shape[0]),:]
    target_test=targ[int(v*inp.shape[0]):,:]

    input_train=to_categorical(input_train,num_classes=numstates)
    target_train=to_categorical(target_train,num_classes=numstates)
    input_test=to_categorical(input_test,num_classes=numstates)
    target_test=to_categorical(target_test,num_classes=numstates)

    i_train=tf.convert_to_tensor(input_train[:int(np.floor(input_train.shape[0]/batch_size)*batch_size),:,:])
    t_train=tf.convert_to_tensor(target_train[:int(np.floor(target_train.shape[0]/batch_size)*batch_size),:,:])
    i_test=tf.convert_to_tensor(input_test[:int(np.floor(input_test.shape[0]/batch_size)*batch_size),:,:])
    t_test=tf.convert_to_tensor(target_test[:int(np.floor(target_test.shape[0]/batch_size)*batch_size),:,:])

    return i_train, t_train, i_test, t_test