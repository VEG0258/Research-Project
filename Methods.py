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

def get_batches(data,batch_size,sequence_length,lag=1):
    if lag == 1:
        print("Using default lag value 1")

    num_batch = int(np.size(data,1)/(sequence_length))
    X=np.zeros((batch_size*num_batch,sequence_length))
    Y=np.zeros((batch_size*num_batch,sequence_length))
    c=0
    
    for f in range(batch_size):
        c=f
        for i in range(num_batch):
            m=i*sequence_length
            if m+sequence_length+lag<data.shape[1]:
                X[c,:]=data[f,m:m+sequence_length]
                Y[c,:]=data[f,m+lag:m+sequence_length+lag]
                c+=batch_size
            
    return [X,Y]

def sample(a, temperature=1.0):
    if temperature == 1.0:
        print("Using default temperature valure 1.0")
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def trans_mat(x,numstates,tau=1):
    if tau == 1:
        print("Using default tau value 1")
    space_size=numstates
    xm = x[0:-tau]
    xp = x[tau:]
    JP=np.histogram2d(xm,xp,bins=np.arange(space_size+1)+1)[0]
    sums = np.sum(JP, axis=1, keepdims=True)  # Sum over each row to normalize
    sums[sums == 0] = 1  # Avoid division by zero; replace 0 sums with 1
    T = JP / sums  # Normalize to get transition probabilities
    T[np.isnan(T)] = 1 / numstates  # Set NaN values if any (should be none with the fix) to 1/numstates

   
    return T.T

def errory_plot(x,y,erry,style,style2,xlabel=None,ylabel=None,limits=None,label=None,title=None,scalex = 'linear',scaley='linear'): 
    plt.plot(x,y,style,label = label)
    plt.fill_between(x,y-erry,y+erry,color=style2)
    if limits:
        plt.axis(limits)
    plt.xscale(scalex)
    plt.yscale(scaley)
    plt.tick_params(labelsize=13)
    if title:
        plt.title(title,fontsize=20)
    if ylabel:
        plt.ylabel(ylabel,fontsize=20)
    if xlabel:
        plt.xlabel(xlabel,fontsize=20)

def evals(T,indices=range(1,6,1)):
    ls = np.absolute(np.linalg.eigvals(T))
    ls=np.sort(ls)[::-1]
    return ls[indices]

def eval_stat(data,tau,num=6,ndata=59):
    x = data
    T = trans_mat(x,tau=tau)
    ls = evals(T)
    sem=np.std(ls,axis=0)/np.sqrt(ndata)
    return ls,sem

def plot(x,y,style,xlabel,ylabel,limits=None,label=None,title=None,scalex = 'linear',scaley='linear'):     
    pylab.plot(x,y,style,label = label)
    if limits:
        pylab.axis(limits)
    else:
        limits=[min(x),max(x),min(y),max(y)]
        pylab.axis(limits)
    pylab.xscale(scalex)
    pylab.yscale(scaley)
    pylab.tick_params(labelsize=13)
    if title:
        pylab.title(title,fontsize=20)
    pylab.ylabel(ylabel,fontsize=20)
    pylab.xlabel(xlabel,fontsize=20)
    
def plot_scatter(points,colors='b',marker=None,f=None,groups=False,annots=None,xlabel=None,ylabel=None,title=None,path=None):
    fig, ax = plt.subplots(figsize=(10,8))
    if groups==False:
        ax.scatter(points[:,0], points[:,1],s=100,c=colors,marker=marker)
    else:
        for group,color in zip(groups,colors):
            sub = points[f == group]
            ax.scatter(sub[:,0], sub[:,1],s=100,c=color,marker=marker)
        
    if annots:
        for annot,point in zip(annots,points):
            ax.annotate(annot, (point[0], point[1]))
    if title:
        plt.title(title,fontsize=20)
    if xlabel:
        plt.xlabel(xlabel,fontsize=20)
    if ylabel:
        plt.ylabel(ylabel,fontsize=20)
    plt.tick_params(labelsize=13)
    if path:
        plt.savefig(path)
    return fig, ax

def run_DIB(set_, numstates, pxy):
    (key,beta,clusters) = set_
    #if not (i%(6*trails)):
        #print(beta)
    f0 = np.random.randint(0, high=clusters, size=numstates)
    f,iyt,H,qy_t,qt,k = DIB(pxy,clusters,f0,beta)
    
    return(key,[H,iyt,k,f])
    
def DIB(pxy,clusters,f0,beta,toler=1e-6,max_iter=1000):
    def calc_J(qt,qy_t,py,beta):
        qyt = qy_t*qt
        H = - (qt[0,qt[0,:]>0]*np.log2(qt[0,qt[0,:]>0])).sum()  
        temp = qyt*np.log2(qy_t/py)    
        iyt = temp[~np.isinf(temp) & ~np.isnan(temp)].sum()
        J = H - beta*iyt
        return J,H,iyt
    
    def calc_q(qt,qy_t,f):
        for t in range(clusters):
            qt[0,t] = px[f==t].sum()
            if qt[0,t]>0:
                qy_t[:,t] = pxy[f==t,:].sum(axis=0) / qt[0,t]        
        return qt,qy_t   
    
    dim_x = pxy.shape[0]
    dim_y = pxy.shape[1]
    px = pxy.sum(axis=1)
    py_x = np.zeros((dim_y,dim_x))
    py_x[:,px!=0] = pxy[px!=0,:].T/px[px!=0]
    py = pxy.T.sum(axis=1,keepdims=True)
    qt = np.zeros((1,clusters))
    qy_t = np.zeros((dim_y,clusters))
    
    qt,qy_t = calc_q(qt,qy_t,f0)
    
    d = np.zeros((dim_x,clusters))
    J_new,H,iyt = calc_J(qt,qy_t,py,beta)
    keepRunning = True
    n=1
    diff_old = 0
    while keepRunning:
        J = J_new
        for t in range(clusters):
            tmp = py_x*np.log2(py_x/qy_t[:,t:t+1])
            tmp[np.isinf(tmp) | np.isnan(tmp)]=0
            d[:,t] = tmp.sum(axis=0)
            
        el = np.log2(qt) - beta*d
        f = el.argmax(axis=1)
        qt,qy_t = calc_q(qt,qy_t,f)
        
        J_new,H,iyt = calc_J(qt,qy_t,py,beta)        
        diff_new = np.abs(J_new-J)
        if diff_new==diff_old or np.abs(J_new-J)<toler or n>max_iter:
            break
        else:
            n+=1
        diff_old = diff_new
    vals = np.unique(f)
    if len(vals)<clusters:
        for i,val in enumerate(vals):
            f[f==val] = i
        qy_t = qy_t[:,vals]
        qt = qt[:,vals]
    return f,iyt,H,qy_t,qt,len(vals)

def find_pareto(X):
    
    N,d = X.shape
    idx = np.zeros((N, ), dtype=bool)
    temp = np.zeros((N, d), dtype=bool)
    
    for i in range(N):
        temp[:]=False
        for j in range(d):
            temp[:,j] = X[i,j]<X[:,j]
            
        if temp.sum(axis=1).max()<d:
            idx[i] = True
    return idx

def RandomState(batch_size=None,loc_c=0,loc_h=0,scale_c=0.01,scale_h=0.01):
    if batch_size==None:
        batch_size = self.batch_size
    input_h = np.random.normal(loc=loc_h,scale = scale_h,size=(1,batch_size,n_neurons))
    input_c = np.random.normal(loc=loc_c,scale = scale_c,size=(1,batch_size,n_neurons))
    initial_state = (input_h,input_c)
    return initial_state