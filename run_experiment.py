import pickle
import logging

from experiment_parser import *

from datasets import pixel_data
from models import LeNet
import predictors

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
EPOCHS = 1000000
TRAIN_RATIO = 0.8
IMG_DIM = 400

trials = experiment.get_trials(list_type='all')
X, Y = pixel_data.create_dataset(trials, binary=True)
print(X.shape, Y.shape)
net = LeNet.LeNet(LR=LR)
print(net)

predictors.train(X, Y, net, LR, EPOCHS, TRAIN_RATIO, IMG_DIM)





print('Opening shell...')
print('')

