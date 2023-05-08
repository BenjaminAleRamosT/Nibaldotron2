import pandas as pd
import numpy as np
import utility as ut

# Save Data


def save_data(Data, Label):
    
    np.savez('train.npz', Data[0], Label[0])
    np.savez('test.npz', Data[1], Label[1])

    return 

# normalize data


def data_norm(x, a = 0.01, b = 0.99):

    x_max = np.max(x)
    x_min = np.min(x)
    x = ( ( ( x - x_min )/( x_max - x_min ) ) * ( b - a ) ) + a
    
    return x

# Binary Label


def binary_label(classes):
    n_class = np.max(classes)
    classes = classes -1

    label = np.zeros( (classes.shape[0],n_class) )
    label[np.arange(0,len(classes)),classes] = 1
    return label


# Load data

def load_data(path_csv = ['train.csv','test.csv']):
    
    data = []
    label = []
    for path in path_csv:
        
        data_class = np.genfromtxt(path, delimiter=',')
        #norm = data_norm(data_class[:,:-1].T)
        data.append(data_class[:,:-1].T)                               #datos 
        label.append(binary_label(data_class[:,-1].astype(int)).T )    #labels
        
    return data,label


# Beginning ...
def main():
    print('Cargando data...')
    Data, Label = load_data()
    print('Guardando data...')
    save_data(Data, Label)
    

if __name__ == '__main__':
    main()
