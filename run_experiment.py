import pickle
import logging

from experiment_parser import *

from datasets import pixel_data
from models import LeNet
from predictors import *


EXP_NAME = 'scale_ranking_bmm_short'
DATA_FOLDER = EXP_NAME + '_data'

PICKLE_FILE = EXP_NAME + '.pik'

RUN_PARSER = False


def parse_and_save():
    experiment = Experiment(EXP_NAME, DATA_FOLDER)
    with open(PICKLE_FILE, 'wb') as f:
        print('Saving file...')
        pickle.dump(experiment, f, -1)
    return experiment


try:
    with open(PICKLE_FILE, 'rb') as f:
        if RUN_PARSER:
            experiment = parse_and_save()
        else:
            print('Opening file: ' + PICKLE_FILE)
            experiment = pickle.load(f)
except FileNotFoundError:
    experiment = parse_and_save()


# Hyperparameters
LR = 1e-3
EPOCHS = 1e6
TRAIN_RATIO = 0.8

trials = experiment.get_trials(list_type='all')
X, Y = pixel_data.create_dataset(trials)
net = LeNet.LeNet(LR=LR)





print('Opening shell...')
print('')

