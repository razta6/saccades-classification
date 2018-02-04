import numpy as np
from experiment_parser import *

def create_dataset(trials):
    num_trials = len(trials)
    X = []
    Y = []

    for i in range(num_trials):
        trial = trials[i].to_array()
        X.append(trial.flatten())
        Y.append(trials[i].stim_type_ind)

    return np.array(X), np.array(Y)
