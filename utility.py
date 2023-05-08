# My Utility : auxiliars functions

import pandas as pd
import numpy as np
import prep as pr

# load config param

def load_config(ruta_ae='cnf_sae.csv',ruta_soft='cnf_softmax.csv'):

    # cnf_sae.csv
    #  Línea 1: Parámetro P-inverse: 100
    #  Línea 2: Func. Activación Encoder: 2
    #  Línea 3: Max. Iteraciones: 100
    #  Línea 4: Tamaño miniBatch: 32
    #  Línea 5: Tasa Aprendizaje: 0.01
    #  Línea 6: Nodos Encoder 1.: 200
    #  Línea 7: Nodos Encoder 2.: 100

    # cnf_softmax.csv
    #  Línea 1: Max. Iteraciones : 200
    #  Línea 2: Tasa Aprendizaje : 0.01
    #  Línea 3: Tamaño miniBatch : 32


    with open(ruta_ae, 'r') as archivo_csv:

        p_sae = [int(i) if '.' not in i else float(i)
                for i in archivo_csv if i != '\n']

    with open(ruta_soft, 'r') as archivo_csv:
    
        p_sft = [int(i) if '.' not in i else float(i)
                for i in archivo_csv if i != '\n']


    return p_sae,p_sft

# Initialize weights for SNN-SGDM
def iniWs(inshape, layer_node):

    W1 = iniW(layer_node, inshape)
    
    W2 = iniW(inshape, layer_node)
    W = list((W1, W2))

    V = []
    for i in range(len(W)):
        V.append(np.zeros(W[i].shape))
        
    return W, V


# Initialize weights for one-layer

def iniW(next, prev):
    r = np.sqrt(6/(next + prev))
    w = np.random.rand(next, prev)
    w = w*2*r-r
    return w

# Feed-forward of SNN


def forward_ae(X, W, Param):
    
    #cambiar activaciones por config
    act_encoder = Param[1]
   
    A = []
    z = []
    Act = []
    
    # data input
    z.append(X)
    A.append(X)
    #print(len(W))
    # iter por la cantidad de pesos
    for i in range(len(W)):
        #print(W[i].shape, X.shape)
        X = np.dot(W[i], X)
        z.append(X)
        if i == 0:
            X = act_function(X, act=act_encoder)
        
        A.append(X)
        
    Act.append(A)
    Act.append(z)

    return Act


# Activation function
def act_function(x, act=1, a_ELU=1, a_SELU=1.6732, lambd=1.0507):

    # Relu

    if act == 1:
        condition = x > 0
        return np.where(condition, x, np.zeros(x.shape))

    # LRelu

    if act == 2:
        condition = x >= 0
        return np.where(condition, x, x * 0.01)

    # ELU

    if act == 3:
        condition = x > 0
        return np.where(condition, x, a_ELU * np.expm1(x))

    # SELU

    if act == 4:
        condition = x > 0
        return lambd * np.where(condition, x, a_SELU * np.expm1(x))

    # Sigmoid

    if act == 5:
        return 1 / (1 + np.exp(-1*x))

    return x


# Derivatives of the activation funciton


def deriva_act(x, act=1, a_ELU=1, a_SELU=1.6732, lambd=1.0507):

    # Relu

    if act == 1:
        condition = x > 0
        return np.where(condition, np.ones(x.shape), np.zeros(x.shape))

    # LRelu

    if act == 2:
        condition = x > 0
        return np.where(condition, np.ones(x.shape), np.ones(x.shape) * 0.01)

    # ELU

    if act == 3:
        condition = x > 0
        return np.where(condition, np.ones(x.shape), a_ELU * np.exp(x))

    # SELU falta

    if act == 4:
        condition = x > 0
        return lambd * np.where(condition, np.ones(x.shape), a_SELU * np.exp(x))

    # Sigmoid

    if act == 5:
        # pasarle la sigmoid
        return np.multiply(act_function(x, act=5) , (1 - act_function(x, act=5)))

    return x

# Calculate Pseudo-inverse
def pinv_ae(x , H , C):     
    
    A = np.dot(H,H.T) + (1/C)
    U , S , V = np.linalg.svd(A)
    
    S_inv = np.diag( 1/ S )
    
    A_inv = np.linalg.multi_dot([V.T , S_inv , U.T])
    
    w2 = np.linalg.multi_dot([x , H.T , A_inv])
    
    return(w2)

# Feed-Backward of SNN
def gradW_ae(Act, W, Param):
    '''
    Act = lista de resultados de cada capa,
    data activada en [0] y no activada en [1]
    '''
    
    act_encoder = Param[1]
    
    L = len(Act[0])-1
    
    M = Param[3]
   
    e = Act[0][L] - Act[0][0]
    
    Cost = np.sum(np.sum(np.square(e), axis=0)/2)/M
    
    # grad decoder
    
    delta = e
    #gW_l = np.dot(delta, Act[0][L-1].T)/M

    #gW.append(gW_l)

    # grad encoder

    t1 = np.dot(W[1].T, delta)

    t2 = deriva_act(Act[1][1], act=act_encoder)

    delta = np.multiply(t1, t2)

    t3 = Act[0][0].T

    gW = np.dot(delta, t3)/M
    
    #gW.append(gW_l)

    #gW.reverse()
    
    return gW, Cost

# Update W and V


def updWV_RMSprop(W, V, gW, tasa =  0.1):

    e = 10**-8
    beta = 0.9
   
    V = (beta * V) + ( (1-beta)* np.square(gW))
    gRMS = np.multiply(1/np.sqrt(V+e),gW)
    W = W - tasa * gRMS
    return W, V

def updWV_sgdm(W, V, gW):

    tasa = 0.01
    beta = 0.8
    # print('ajuste')
    for i in range(len(W)): 
        
        V[i] = (beta * V[i]) + (tasa*gW[i])
        W[i] = W[i] - V[i]
        W[i] = pr.data_norm(W[i])

    return W, V

# Softmax's gradient
def gradW_softmax(x,y,a):
    
    
    
    M      = y.shape[1]
    Cost   = -(np.sum(np.sum(  np.multiply(y,np.log(a)) , axis=0)/2))/M
    gW     = -(np.dot(y-a,x.T))/M
    return gW, Cost

# Calculate Softmax
def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return(exp_z/exp_z.sum(axis=0,keepdims=True))

#Save weights and MSE  of the SNN
def save_w_dl(W,Ws,Cost):
    np.savez('wAEs.npz', W[0], W[1])
    np.savez('wSoftmax.npz', Ws)
    
    
    df = pd.DataFrame( Cost )
    df.to_csv('costo.csv',index=False, header = False )
    
    return

# -----------------------------------------------------------------------






