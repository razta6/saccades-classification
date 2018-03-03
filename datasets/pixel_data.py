import numpy as np
from experiment_parser import *


def create_dataset(trials, label_set, binary=True):
    num_trials = len(trials)
    X = []
    Y = []
    
    print('Building dataset...')
    for i in range(num_trials):
        if (i+1)%100==0:
            print(i+1)
        trial = trials[i].to_array()
        X.append(trial.flatten())
        #Y.append(trials[i].stim_type_ind)
        Y.append(trials[i].get_bid(bid_type='binary'))
    
    print('Done')
    print()
    
    X, Y = np.array(X), np.array(Y)
    
    for i in range(len(label_set)):
        Y[Y==(label_set[i])] = i
        
    idx = np.where(Y<len(label_set))
    X = X[idx]
    Y = Y[idx]

    return X, Y
