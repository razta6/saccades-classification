import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


TRIAL_START = 'TrialStart'
TRIAL_END = 'ScaleStart'

SCREEN_HEIGHT = 1080
SCREEN_WIDTH = 1920

STIM_HEIGHT = 400
STIM_WIDTH = 400

MIN_SAMPLES_PER_TRIAL = 10


class Experiment:
    """
        parse and combine the data from eyelink recordings (edf parsed to asc) and behavioral data (txt)
        assume 'data' folder has 'asc' folder and 'txt' folder
        each folder contains for every subject the same file name with different extension
        asc foldr constructs the subjects, therefore it is run before txt folder
        this may lead to missing subjects that have behavioral data (txt) but no eyelink files (asc)
    """
    def __init__(self, name='', data=''):
        self.name = name
        self.data = data
        self.subjects = []
        self.stim_list = self.create_stim_list()
        self.valid = False
        self.error = ''

        self.logger_init()
        self.parse_folder()

        self.exp_post_process()

        if not self.valid:  # it's not an else statement because of the cluster extraction
            logging.warning('EXPERIMENT INVALID')
            logging.info(self.error)

        self.log_summary()

    def logger_init(self):
        log_fname = self.name + '.log'
        logging.basicConfig(filename=log_fname, filemode='w', level=logging.DEBUG)
        logging.info('Started')

    def add_subject(self, subject):
        self.subjects.append(subject)

    def remove_subject(self, subject):
        new = [sub for sub in self.subjects if sub.get_id() != subject.get_id()]
        self.subjects = new

    # TODO: getters and setters
    def get_subjects(self, just_valids=True):
        if just_valids:
            return [subject for subject in self.subjects if subject.valid]
        return self.subjects

    def get_num_subjects(self, just_valids=True):
        return len(self.get_subjects(just_valids))

    def parse_folder(self):
        # TODO: do matchings between txt and asc folder and invalidate subjects that don't have both

        for ext in ['asc', 'txt']:
            print('Experiment name: ' + self.name)
            logging.info('Experiment name: ' + self.name)

            logging.info('Folder name: ' + self.data)
            print('File type: ' + ext)
            logging.info('File type: ' + ext)
            output_folder = os.path.join(self.data, 'output')
            folder = os.path.join(output_folder, ext)
            for fname in os.listdir(folder):
                if not fname.endswith(ext):
                    continue
                fname_list = fname.split('_')
                try:
                    id = int(fname_list[2])
                except ValueError:  # wrong file name
                    continue
                if ext == 'asc':
                    subject = Subject(id, fname, self.data)
                    print('Parsing subject number %d' % (self.get_num_subjects(False) + 1))
                    logging.info('Parsing subject number %d' % (self.get_num_subjects(False) + 1))
                else:
                    subject = self.get_subject_by_id(id)
                    if subject is not None:
                        print('Updating subject %d' % (subject.get_id()))
                        logging.info('Updating subject %d' % (subject.get_id()))
                if subject is None:
                    continue  # TODO: deal with it
                path = os.path.join(folder, fname)
                if ext == 'asc':
                    subject.parse_asc_file(path)
                    # if subject.valid:  # TODO: should add also invalid subjects and deal with it
                    self.add_subject(subject)
                else:
                    subject.parse_txt_file(path)
                    if not subject.valid:
                        self.error += 'Subject ' + str(subject.get_id()) + ' is invalid and removed from experiment'
                        # self.remove_subject(subject)  # TODO: should add also invalid subjects and deal with it
            print('Finished ' + ext + ' folder')
            logging.info('Finished ' + ext + ' folder')

        self.valid = True

    def get_subject_by_id(self, id):
        subjects = self.get_subjects()
        for subject in subjects:
            if subject.get_id() == id:
                return subject
        return None

    def get_trials(self, list_type='all'):
        result = []
        subjects = self.get_subjects()

        if list_type == 'all':
            for subject in subjects:
                result.extend(subject.get_trials())
        if list_type == 'subject':
            for subject in subjects:
                result.append(subject.get_trials())
        if list_type == 'stim':
            pass

        return result

    def exp_post_process(self):
        subjects = self.get_subjects()
        for subject in subjects:
            valid = subject.subject_post_process()
            if not valid:
                logging.warning('Invalid subject: ' + subject.get_error() + ' (subject = %d)' % subject.get_id())

        subjects = self.get_subjects()
        if len(subjects) == 0:
            self.valid = False
            self.error += 'Experiment has no valid subjects after post-processing.'

        return self.valid

    def create_stim_list(self):
        print('Creating stim list...')
        stims = []
        ext = 'jpg'
        stim_folder = 'stim'
        folder = os.path.join(self.data, stim_folder)
        for fname in os.listdir(folder):
            if not fname.endswith(ext):
                continue
            stims.append(fname)
        return stims

    def log_summary(self):
        logging.info('-------------------------------------')
        logging.info('Experiment processing summary:')
        valid_num_of_subjects = self.get_num_subjects()
        invalid_num_of_subjects = self.get_num_subjects(False)
        logging.info('Number of subjects: %d/%d valids' % (valid_num_of_subjects, invalid_num_of_subjects))

        # get all trials (only from valid subjects)
        valid_num_of_trials = 0
        invalid_num_of_trials = 0
        subjects_string = ''
        for subject in self.get_subjects():
            valid_num_of_trials += subject.get_num_trials()
            invalid_num_of_trials += subject.get_num_trials(False)
            subjects_string += 'Subject ID: %d, Valid trials: %d/%d' % (subject.get_id(), subject.get_num_trials(), subject.get_num_trials(False)) + '\n'

        logging.info('Number of trials: %d/%d valids' % (valid_num_of_trials, invalid_num_of_trials))
        logging.info(subjects_string)


class Subject:

    def __init__(self, id=0, fname='', data_folder=''):
        self.id = id
        self.fname = fname
        self.data_folder = data_folder
        self.trials = []
        self.valid = False
        self.error = ''

    def add_trial(self, trial):
        self.trials.append(trial)

    # TODO: getters and setters
    def get_trials(self, just_valids=True):
        if just_valids:
            return [trial for trial in self.trials if trial.valid]
        return self.trials

    def get_id(self):
        return self.id

    def get_num_trials(self, just_valids=True):
        return len(self.get_trials(just_valids))

    def get_trial_by_num(self, num):
        for trial in self.get_trials(False):
            if trial.trial_num == num:
                return trial
        return None

    def get_trial_by_stim(self, stim):
        for trial in self.get_trials(False):
            if trial.get_stim() == stim:
                return trial
        return None

    def find_trial(self, onsettime):
        eps = 0.5
        trials = self.trials
        for trial in trials:
            diff = trial.get_start_time() - onsettime
            if np.absolute(diff) < eps:
                return trial
        return None

    def get_error(self):
        return self.error

    # TODO: run on files and parse the file by it extension

    # TODO: make sure first asc files are parsed, maybe by different folsers?

    def parse_asc_file(self, path):
        ext = 'asc'
        with open(path) as f:
            parser = LineParser(ext)
            trial = None

            for line in f:

                if line.startswith('MSG'):
                    ts, flag = parser.parse_msg(line)
                    if not parser.is_int(ts):
                        # TODO: should be 'continue' and find a way to deal with invalid trials
                        break  # something went wrong
                    if len(flag) < 5:
                        continue

                    if flag[0] == TRIAL_START:

                        trial = Trial(self.id, flag, int(ts), self.data_folder)
                        if not trial.init:
                            # TODO: should be 'continue' and find a way to deal with invalid trials
                            break  # something went wrong
                        continue

                    if flag[0] == TRIAL_END:
                        trial.trial_closure(int(ts), flag)
                        trial.valid = self.check_validity()
                        if not trial.valid:
                            # TODO: should be 'continue' and find a way to deal with invalid trials
                            pass  # something went wrong
                        self.add_trial(trial)
                        continue
                elif trial is not None:
                    if trial.init and not trial.valid:  # we want to parse xy positions
                        curr = line.split()

                        if not parser.is_position(curr):  # not a position line
                            continue

                        ts, x, y = int(curr[0]), float(curr[1]), float(curr[2])
                        y = SCREEN_HEIGHT - y  # flip y direction so it can be plotted with (0,0) in the bottom left
                        sample = Sample(ts, x, y)
                        trial.add_sample(sample)

            self.valid = self.check_validity()

    def parse_txt_file(self, path):
        ext = 'txt'
        with open(path) as f:
            parser = LineParser(ext)
            subject_check = False  # check once if the subject in the file corresponds to our subject

            for line in f:
                if line.startswith('bmem_short'):
                    id, msg = parser.parse_txt_line(line)
                    if not subject_check:
                        if parser.is_int(id):
                            id = int(id)
                            if id == self.get_id():
                                subject_check = True
                            else:
                                self.error += 'Wrong txt file for subject ' + str(id)
                                self.valid = False
                                break
                        else:
                            continue  # something fishy, skip this line

                    if not parser.is_float(msg[0]):
                        continue

                    trial = self.find_trial(float(msg[0]))
                    if trial is None:
                        continue

                    trial.set_stim(msg[1])

                    if parser.is_float(msg[2]):
                        trial.set_bid(float(msg[2]))

                    if parser.is_float(msg[3]):
                        trial.set_rt(float(msg[3]))

                    trial.set_stim_type(msg[4])

                    if parser.is_int(msg[5]):
                        trial.set_stim_type_ind(int(msg[5]))

    def check_validity(self):
        return True

    def subject_post_process(self):
        trials = self.get_trials()
        for trial in trials:
            valid = trial.trial_post_process()
            if not valid:
                logging.warning('Invalid trial: ' + trial.get_error() + ' (trial = %d)' % trial.get_num())

        trials = self.get_trials()
        if len(trials) == 0:
            self.valid = False
            self.error += 'Subject has no trials after post-processing.'

        if not self.adjust_bids():
            self.valid = False
            self.error += 'Error processing adjusted bids'

        if not self.make_bids_binary():
            self.valid = False
            self.error += 'Error processing binary bids'

        return self.valid

    def adjust_bids(self):
        trials = self.get_trials()
        bids = np.array([trial.get_bid() for trial in trials])
        mean_bid = np.mean(bids)
        # adjust all bids by the mean
        for trial in trials:
            trial.adjusted_bid = trial.get_bid() - mean_bid

        # sanity check
        bids = np.array([trial.get_bid('adjusted') for trial in trials])
        eps = 1e-2
        if np.abs(np.mean(bids)) > eps:
            return False
        return True

    def make_bids_binary(self):
        trials = self.get_trials()
        for trial in trials:
            if np.sign(trial.get_bid('adjusted')) < 0:
                trial.binary_bid = 0
            else:
                trial.binary_bid = 1

        # sanity check
        binary_bids = np.array([trial.get_bid('binary') for trial in trials])
        zeros = np.count_nonzero(binary_bids == 0)
        ones = np.count_nonzero(binary_bids == 1)
        if zeros + ones != len(binary_bids):
            return False
        return True


class Trial:

    def __init__(self, subject_id, flag, ts=0, data_folder=''):
        self.subject_id = subject_id
        self.trial_num = 0
        self.start_time = 0.0
        self.start_ts = ts
        self.end_time = 0.0
        self.end_ts = ts
        self.rt = 0.0
        self.sample_rate = 0  # ms between two samples

        self.stim = ''
        self.data_folder = data_folder

        self.bid = -1
        self.adjusted_bid = -11
        self.binary_bid = -1

        self.stim_type = ''
        self.stim_type_ind = 0

        self.samples = []
        self.clusters = []  # to be updated after clustering

        self.extracted_clusters = False
        self.init = False
        self.valid = False
        self.error = ''

        if self.trial_init(flag) is not None:
            self.init = True

    def add_sample(self, sample):
        self.samples.append(sample)

    def add_cluster(self, cluster):
        self.clusters.append(cluster)

    def tolist(self, just_valids=True):
        result = []
        for sample in self.get_samples(just_valids):
            ts = sample.ts
            x, y = sample.xy()
            result.append([ts, x, y])
        return result

    def to_array(self, just_valids=True):
        sample_array = np.zeros((STIM_HEIGHT, STIM_WIDTH))
        samples = self.tolist(just_valids)
        for sample in samples:
            x = sample[1] - (SCREEN_WIDTH/2) + (STIM_WIDTH/2)
            y = sample[2] - (SCREEN_HEIGHT/2) + (STIM_HEIGHT/2)
            sample_array[int(x)][int(y)] += 1
        return sample_array

    # TODO: getters and setters
    def get_samples(self, just_valids=True):
        if just_valids:
            return [sample for sample in self.samples if sample.is_valid()]
        return self.samples

    def get_clusters(self, just_valids=True):
        if just_valids:
            newlist = [cluster for cluster in self.clusters if cluster.is_valid()]
            return sorted(newlist, key=lambda x: x.start)
        return sorted(self.clusters, key=lambda x: x.start)

    def get_cluster_by_num(self, num):
        for cluster in self.get_clusters(False):
            if cluster.num == num:
                return cluster
        return None

    def get_subject_id(self):
        return self.subject_id

    def get_num(self):
        return self.trial_num

    def get_start_time(self):
        return self.start_time

    def get_num_samples(self, just_valids=True):
        return len(self.get_samples(just_valids))

    def get_num_clusters(self, just_valids=True):
        return len(self.get_clusters(just_valids))

    def get_stim(self):
        return self.stim

    def get_bid(self, bid_type='original'):
        if bid_type == 'original':
            return self.bid
        if bid_type == 'adjusted':
            return self.adjusted_bid
        if bid_type == 'binary':
            return self.binary_bid
        return None

    def get_error(self):
        return self.error

    def set_stim(self, stim):
        self.stim = stim

    def set_bid(self, bid):
        self.bid = bid

    def set_rt(self, rt):
        self.rt = rt

    def set_stim_type(self, stim_type):
        self.stim_type = stim_type

    def set_stim_type_ind(self, stim_type_ind):
        self.stim_type_ind = stim_type_ind

    # parse list ['ScaleStart', 'TaskBDM', 'Run001', 'Trial001', 'Time2.1339']
    def trial_init(self, flag):
        if flag[0] != TRIAL_START:
            return None
        trial_num_as_str = flag[3].replace('Trial', '')
        self.trial_num = int(trial_num_as_str)
        trial_time_as_str = flag[4].replace('Time', '')
        self.start_time = float(trial_time_as_str)
        self.end_time = float(trial_time_as_str)
        return True

    # parse list ['ScaleStart', 'TaskBDM', 'Run001', 'Trial001', 'Time2.1339']
    def trial_closure(self, ts, flag):
        if flag[0] != TRIAL_END:
            return None
        curr_trial_num = self.trial_num
        trial_num_as_str = flag[3].replace('Trial', '')
        trial_num = int(trial_num_as_str)
        if curr_trial_num != trial_num:
            self.valid = False
            return None
        trial_time_as_str = flag[4].replace('Time', '')
        trial_time = float(trial_time_as_str)
        self.end_time = trial_time
        self.end_ts = ts
        self.valid = True

        return True

    '''
    def check_validity(self):
        if self.get_num_samples() < MIN_SAMPLES_PER_TRIAL:
            return False
        return True
    '''

    def calc_sample_rate(self):
        if self.get_num_samples() < 2:
            return False

        samples = self.get_samples()
        ts = [sample.ts for sample in samples]

        # calc trial ts by the majority of distances
        diffs = np.array([j-i for i,j in zip(ts[:-1], ts[1:])])  # calc the diff between each two consecutive timestamps
        bins = np.bincount(diffs)  # count the number of times each diff is in diffs
        most_common_diff = np.argmax(bins)  # return the value oof the most common diff in diffs

        if most_common_diff == 0 or most_common_diff > 10:  # which can be considered as valid sample rate
            return False
        self.sample_rate = most_common_diff

        return True

    def trial_post_process(self):
        samples = self.get_samples()
        for sample in samples:
            valid = sample.sample_post_process()
            if not valid:
                logging.warning('Invalid sample: ' + sample.get_error() + ' (ts = %d)' % sample.ts)
        samples = self.get_samples()

        # TODO: remove pre\post blink samples

        if len(samples) < MIN_SAMPLES_PER_TRIAL:
            self.valid = False
            self.error += 'Trial has no samples after post-processing.'

        if self.bid < 0 or self.bid > 10:
            self.valid = False
            self.error += 'Trial bid is out of range (0-10).'

        if self.extracted_clusters:
            if self.get_num_clusters() == 0:
                self.valid = False
                self.error += 'Trial has no clusters after postprocessing.'

        if self.valid:
            valid = self.calc_sample_rate()
            if not valid:
                self.valid = False
                self.error += 'Trial sample rate is invalid.'

        return self.valid

    def plot(self, marker=',', linestyle='None', rect=True, just_valids=True):
        print('Plotting...')
        trial_list = np.array(self.tolist(just_valids))  # get first trial, as a list of [ts, x, y]
        X = trial_list[:, 1]
        Y = trial_list[:, 2]
        plt.plot(X, Y, marker=marker, linestyle=linestyle)  # plot a scatter of the x,y samples
        plt.xlim((0, SCREEN_WIDTH))
        plt.ylim((0, SCREEN_HEIGHT))
        if rect:
            ax = plt.gca()
            ax.add_patch(Rectangle(((SCREEN_WIDTH/2)-(STIM_WIDTH/2), (SCREEN_HEIGHT/2)-(STIM_HEIGHT/2)), STIM_WIDTH, STIM_HEIGHT, facecolor="grey"))
        plt.show()

    def show_stim(self, plot=True, marker=',', linestyle='None'):
        print('Opening stim...')
        stim_dir = 'stim'
        folder = os.path.join(self.data_folder, stim_dir)
        path = os.path.join(folder, self.stim)
        img = plt.imread(path)
        plt.imshow(img)
        trial_list = np.array(self.tolist())  # get first trial, as a list of [ts, x, y]
        X = trial_list[:, 1]
        Y = trial_list[:, 2]
        Y = SCREEN_HEIGHT - Y  # flip Y values to be like original recordnigs
        # relocate the samples to fit on the image
        X = X - (SCREEN_WIDTH/2) + (STIM_WIDTH/2)
        Y = Y - (SCREEN_HEIGHT/2) + (STIM_HEIGHT/2)
        plt.plot(X, Y, marker=marker, linestyle=linestyle)  # plot a scatter of the x,y samples
        plt.show()


class Sample:

    def __init__(self, ts=0, x=0.0, y=0.0):
        self.ts = ts
        self.x = x
        self.y = y
        self.cluster = 0  # to be updated after clustering
        self.valid = True
        self.error = ''

    def is_valid(self):
        return self.valid

    def xy(self):
        return self.x, self.y

    def get_error(self):
        return self.error

    def sample_post_process(self):
        # check if the sample is on the stimulus
        x = self.x - (SCREEN_WIDTH/2) + (STIM_WIDTH/2)
        y = self.y - (SCREEN_HEIGHT/2) + (STIM_HEIGHT/2)
        if x < 0 or x >= STIM_WIDTH:
            self.valid = False
            self.error += 'Sample x dim is outside stimulus width.'
        if y < 0 or y >= STIM_HEIGHT:
            self.valid = False
            self.error += 'Sample y dim is outside stimulus height.'
        return self.valid


class LineParser:

    def __init__(self, ext=''):
        self.ext = ext  # asc, txt

    def parse_msg(self, line):
        msg = line.split()
        flag = msg[2].split('_')
        return msg[1], flag[1:]

    def parse_txt_line(self, line):
        msg = line.split()
        id_str = msg[0].split('_')
        return id_str[-1], msg[2:]

    def is_int(self, num):
        try:
            int(num)
            return True
        except ValueError:
            return False

    def is_float(self, num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    # check if line is a position line, [388128, 935.5, 508.9, 3036, 0, ...]
    def is_position(self, line):
        if len(line) == 0:  # empty line
            return False
        if len(line) < 4:  # not a position, safety for next condition
            return False
        if not (self.is_int(line[0]) and self.is_float(line[1]) and self.is_float(line[2])):  # not a position or no recordings
            return False
        return True

