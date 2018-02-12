import pickle
import logging

from experiment_parser import *
from plotters import heatmap_plotter

from datasets import pixel_data, heatmap_data
from models import LeNet
from predictors import binary_classificaion, multiclass_classification

EXP_NAME = 'scale_ranking_bmm_short'
DATA_FOLDER = EXP_NAME + '_data'

PICKLE_FILE = EXP_NAME + '.pik'

RUN_PARSER = False
RUN_PLOTTER = False

BINARY = True
MULTI = False

PIXEL_DATA = False
HEATMAP_DATA = True

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

if RUN_PLOTTER:
    trials = experiment.get_trials(list_type='all')
    real_labels = [trial.stim_type_ind for trial in trials]
    labels = heatmap_plotter.run_ploter(trials, save=True, plot=False)
    for i in range(len(trials)):
        if real_labels[i]!=labels[i]:
            print('i = %d, real = %d, fake = %d' % (i, real_labels[i], labels[i]))


# Hyperparameters
LR = 1e-3
EPOCHS = 1000000
TRAIN_RATIO = 0.8
IMG_DIM = 400

trials = experiment.get_trials(list_type='all')

if BINARY:
    if PIXEL_DATA:
        X, Y = pixel_data.create_dataset(trials, binary=True)
    if HEATMAP_DATA:
        X, Y = heatmap_data.create_dataset(trials, binary=True)
    print(X.shape, Y.shape)
    net = LeNet.LeNet_binary(LR=LR)
    print(net)
    print('BINARY')
    binary_classificaion.train(X, Y, net, LR, EPOCHS, TRAIN_RATIO, IMG_DIM, heatmap=HEATMAP_DATA)

if MULTI:
    if PIXEL_DATA:
        X, Y = pixel_data.create_dataset(trials, binary=False)
    if HEATMAP_DATA:
        X, Y = heatmap_data.create_dataset(trials, binary=False)
    print(X.shape, Y.shape)
    net = LeNet.LeNet_multiclass(LR=LR)
    print(net)
    print('MULTICLASS')
    multiclass_classification.train(X, Y, net, LR, EPOCHS, TRAIN_RATIO, IMG_DIM, heatmap=HEATMAP_DATA)




print('Opening shell...')
print('')

