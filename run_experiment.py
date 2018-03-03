import pickle
import logging

import experiment_parser
import bdm_experiment_parser
from plotters import heatmap_plotter

from datasets import pixel_data, heatmap_data
from models import LeNet
from predictors import binary_classificaion, multiclass_classification

EXP_NAME = 'scale_ranking_bmm_short'
EXP_NAME2 = 'bdm_bmm_short'

DATA_FOLDER = EXP_NAME + '_data'
DATA_FOLDER2 = EXP_NAME2 + '_data'

PICKLE_FILE = EXP_NAME + '.pik'
PICKLE_FILE2 = EXP_NAME2 + '.pik'

USE_EXP1 = True
USE_EXP2 = False

RUN_PARSER = False
RUN_PLOTTER = False

BINARY = True
MULTI = False
NUM_OF_CLASSES = 4

PIXEL_DATA = True
HEATMAP_DATA = False

def parse_and_save(exp, data, pik):
    if exp[0]=='s':
        experiment = experiment_parser.Experiment(exp, data)
    else:
        experiment = bdm_experiment_parser.Experiment(exp, data)
    with open(pik, 'wb') as f:
        print('Saving file...')
        pickle.dump(experiment, f, -1)
        print('finished saving')
    return experiment

if USE_EXP1:
    try:
        with open(PICKLE_FILE, 'rb') as f:
            if RUN_PARSER:
                experiment = parse_and_save(EXP_NAME, DATA_FOLDER, PICKLE_FILE)
            else:
                print('Opening file: ' + PICKLE_FILE)
                experiment = pickle.load(f)
    except FileNotFoundError:
        experiment = parse_and_save(EXP_NAME, DATA_FOLDER, PICKLE_FILE)


if USE_EXP2:
    try:
        with open(PICKLE_FILE2, 'rb') as f:
            if RUN_PARSER:
                experiment2 = parse_and_save(EXP_NAME2, DATA_FOLDER2, PICKLE_FILE2)
            else:
                print('Opening file: ' + PICKLE_FILE2)
                experiment2 = pickle.load(f)
    except FileNotFoundError:
        experiment2 = parse_and_save(EXP_NAME2, DATA_FOLDER2, PICKLE_FILE2)


if RUN_PLOTTER:
    trials = experiment2.get_trials(list_type='all')
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

trials = experiment.get_trials(list_type='all') #+ experiment2.get_trials(list_type='all')
#trials = [trial for trial in trials if trial.stim_type_ind==2]

if BINARY:
    label_set = [0,1]
    if PIXEL_DATA:
        X, Y = pixel_data.create_dataset(trials, label_set, binary=True)
    if HEATMAP_DATA:
        X, Y = heatmap_data.create_dataset(trials, binary=True)
    print(X.shape, Y.shape)
    print("Dataset distribution:    Y==0: %.2f, Y==1: %.2f" % (len(Y[Y==0])/len(Y), len(Y[Y==1])/len(Y)))
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
    net = LeNet.LeNet_multiclass(LR=LR, num_of_classes=NUM_OF_CLASSES)
    print(net)
    print('MULTICLASS')
    multiclass_classification.train(X, Y, net, LR, EPOCHS, TRAIN_RATIO, IMG_DIM, heatmap=HEATMAP_DATA)




print('Opening shell...')
print('')

