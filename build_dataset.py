import pickle
import logging

from experiment_parser import *


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


subjects = experiment.get_subjects()
stims = experiment.stim_list

sub = 0
stim = 0

print('Opening shell...')
print('')

