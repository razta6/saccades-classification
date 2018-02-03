import numpy as np
from experiment_parser import *

def create_dataset(trials):
    num_trials = len(trials)
    X = np.zeros(num_trials)
    Y = np.zeros(num_trials)

    for i in range(num_trials):
        trial = trials[i].to_array()
        x_len = trial.shape[0] * trial.shape[1]
        X[i] = np.reshape(trial, x_len)
        Y[i] = trials[i].stim_type_ind

    return X, Y


