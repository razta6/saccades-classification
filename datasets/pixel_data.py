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
        Y.append(trials[i].stim_type_ind)
    
    print('Done')
    print()
    
    X, Y = np.array(X), np.array(Y)
    
    Y[Y==2] = 0
    Y[Y==3] = 1
    Y[Y==6] = 2
    
    if binary:
        idx = np.where(Y!=2)
        print(idx)
        X = X[idx]
        Y = Y[idx]
        
    return X, Y
