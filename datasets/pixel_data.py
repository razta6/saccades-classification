import numpy as np
from experiment_parser import *

def create_dataset(trials, binary=True):
    num_trials = len(trials)
    X = []
    Y = []
    
    print('Building dataset...')
    for i in range(num_trials):
        if (i+1)%100==0:
            print(i+1)
        trial = trials[i].to_array()
        X.append(trial.flatten())
        #Y.append(trials[i].stim_type_ind*10)
        Y.append(trials[i].get_bid(bid_type='binary'))
    
    print('Done')
    print()
    
    X, Y = np.array(X), np.array(Y)
    '''
    Y[Y==(1*10)] = 0
    Y[Y==(2*10)] = 1
    Y[Y==(3*10)] = 2
    Y[Y==(6*10)] = 3
    
    if binary:
        Y[Y==(0)] = 10
        Y[Y==(1)] = 11
        Y[Y==(2)] = 12
        Y[Y==(3)] = 13

        Y[Y==(10)] = 0
        Y[Y==(11)] = 1
        
        idx = np.where(Y<2)
        X = X[idx]
        Y = Y[idx]
    ''' 
    return X, Y
