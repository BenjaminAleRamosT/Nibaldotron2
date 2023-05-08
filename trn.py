# SNN's Training :

import pandas as pd
import numpy as np
import utility as ut


# gets Index for n-th miniBatch
def get_Idx_n_Batch(n, M):

    Idx = (n*M, (n*M)+M)

    return(Idx)  # tuple de indices


# Training miniBatch for softmax
def train_sft_batch(X, Y, W, V, Param):

    costo = []
    M = Param[2]
    numBatch = np.int16(np.floor(X.shape[1]/M))

    for n in range(numBatch):
        Idx = get_Idx_n_Batch(n, M)
        xe, ye = X[:,slice(*Idx)], Y[:,slice(*Idx)]
        
        
        z = np.dot(W, xe)
        
        a = ut.softmax(z)
        gW, Cost = ut.gradW_softmax(xe, ye, a)
        W, V = ut.updWV_RMSprop(W, V, gW, tasa=Param[1])

        costo.append(Cost)

    return(W, V, costo)

# Softmax's training via SGD with Momentum


def train_softmax(X, Y, Param):
    
    W = ut.iniW(Y.shape[0], X.shape[0])
    V = np.zeros(W.shape)
    #X, Y = sort_data_ramdom(X, Y)
    Cost = []
    for Iter in range(1,Param[0]+1):
        idx   = np.random.permutation(X.shape[1])
        xe,ye = X[:,idx],Y[:,idx]   
        
        W, V, c = train_sft_batch(xe, ye, W, V, Param)

        Cost.append(np.mean(c))
        
        if Iter % 10 == 0:
            print('\tIterar-SoftMax: ', Iter, ' Cost: ', Cost[Iter-1])

    return W, Cost


# sort data random
def sort_data_ramdom(X, Y):

    return np.random.shuffle(X.T), np.random.shuffle(Y.T)

# AE's Training with miniBatch


def train_ae_batch(X, W, v, Param):
    
    numBatch = np.int16(np.floor(X.shape[1]/Param[3]))
    
    cost = []
    for n in range(numBatch):
        Idx = get_Idx_n_Batch(n, Param[3])
        xe= X[:,slice(*Idx)]
        
        Act = ut.forward_ae(xe, W, Param)
        
        gW, Cost = ut.gradW_ae(Act, W, Param)
        
        W[0], v[0] = ut.updWV_RMSprop(W[0], v[0], gW[0], Param[4])
        
        W[1] = ut.pinv_ae(xe,Act[0][1],Param[0])
        
        cost.append(Cost)
    return W, v, cost

# AE's Training by use miniBatch RMSprop+Pinv


def train_ae(X, ae_layers, Param):
    
    W, v = ut.iniWs(X.shape[0], ae_layers)
    
    Cost = []
    for Iter in range(1,Param[2]+1):
        #print('Iteracion: ', Iter)
        xe = X[:, np.random.permutation(X.shape[1])]  # sort random
        
        W, v, c = train_ae_batch(xe, W, v, Param)
        Cost.append(np.mean(c))
        if Iter % 10 == 0:
            print('\tIterar-AE: ', Iter, ' Cost: ', Cost[Iter-1])

    return W

# SAE's Training


def train_sae(X, Param):

    W = []
    NumAe = Param[5:]
    for i,ae_layers in enumerate(NumAe):
        print('AutoEncoder: ',i+1,' Capas: ', ae_layers )
        
        w1 = train_ae(X, ae_layers, Param)[0]
        X = ut.act_function(np.dot(w1, X), act = Param[1])
        W.append(w1)

    return W, X

# Load data to train the SNN


def load_data_trn(ruta_trn='train.npz'):
    
    trn = np.load(ruta_trn)
    
    return [trn[i] for i in trn.files]



# Beginning ...
def main():
    print('Cargando config...')
    p_sae, p_sft = ut.load_config()
    print('Cargando data...')
    xe, ye = load_data_trn()
    print('Entrenando ae...')
    W, Xr = train_sae(xe, p_sae)
    print('Entrenando softmax...')
    Ws, cost = train_softmax(Xr, ye, p_sft)
    print('Guardando pesos...')
    ut.save_w_dl(W, Ws, cost)

if __name__ == '__main__':
    main()
