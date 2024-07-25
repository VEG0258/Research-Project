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

def Prediction (dt, model, numFlies, lag, batch_size, numstates):
    #this cell generates and samples the sequences generated from the RNN   
    s_=np.zeros((numFlies,lag*1500,numstates))
    s=np.zeros((numFlies,lag*1500))

    x,_=Methods.get_batches(dt,batch_size,sequence_length)
    x=to_categorical(x,num_classes=numstates)

    tmp0=model.predict(x[:batch_size*100,:,:],batch_size=batch_size)
    for i in range(500):

        tmp=model.predict(to_categorical(tmp0.argmax(axis=2),num_classes=numstates),batch_size=batch_size)
        tmp0=copy(tmp)
        
        s_[:,i*lag:i*lag+lag,:]=copy(tmp[:batch_size,-lag:,:])
        
        if np.mod(i,100)==0:

            f = h5py.File("s_prog.hdf5", "w")
            dset = f.create_dataset("s_", (s_.shape[0],s_.shape[1],s_.shape[2]), data=s_)
            f.close()
            savemat('tmp_prog.mat',{'tmp':tmp})
            model.save_weights('weights.h5')
            print(i)
            
    s=np.zeros((numFlies,300000))
    for i in range(numFlies):
        for t in range(300000):
            s[i,t]=Methods.sample(s_[i,t,:])

    return s_, s