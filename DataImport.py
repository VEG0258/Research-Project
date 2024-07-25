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

def DataImport(name_mat, states, numFlies, numstates):
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

    return dt
