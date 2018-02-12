import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from experiment_parser import *

SCREEN_HEIGHT = 1080
SCREEN_WIDTH = 1920

STIM_HEIGHT = 400
STIM_WIDTH = 400

def run_ploter(trials, save=True, plot=False):
    i = 0
    labels = []
    
    for trial in trials:
        i += 1
        print(str(i) + '/' + str(len(trials)))
        xs, ys = plot_trial_heatmap(trial, save=save, plot=plot)
        labels.append(trial.stim_type_ind)
        
    labels = np.array(labels)
    np.savetxt("labels.csv", labels, delimiter=",")
    
    return labels
        
        
def plot_trial_heatmap(trial, save, plot):
    samples = np.array(trial.tolist())
    xs = samples[:,1]
    ys = samples[:,2]
    
    # Set up the figure
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    fig.set_size_inches(5.21, 5.21)  # hack the size to be saved as 400X400 img (png is saved with 77.5 dpi)
    
    # Draw the two density plots
    ax = sns.kdeplot(xs, ys, n_levels=30, cmap="Greys", shade=True, shade_lowest=False)
    
    # Remove X,Y axis, ticks and labels
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    # Set the plot size to the stimulus size
    xlim, ylim = calc_xy_lim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Remove plot spines (black border)
    for spine in ['top', 'bottom', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    if save:
        folder_name = 'heatmap_plots'
        ext = '.png'
        filename = str(trial.get_subject_id())
        filename += '_' + str(trial.get_num())
        filename += '_' + str(trial.stim_type_ind)
        filename += ext
        path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', folder_name))
        path = os.path.join(path, filename)
        plt.savefig(path, bbox_inches='tight', pad_inches = 0, dpi=100)
        
    if plot:
        plt.show()

    plt.close()
    
    return xs, ys
    
def calc_xy_lim():
    mid_x = SCREEN_WIDTH / 2
    mid_y = SCREEN_HEIGHT / 2

    half_stim_x = STIM_WIDTH / 2
    half_stim_y = STIM_HEIGHT / 2

    xlim = [mid_x - half_stim_x, mid_x + half_stim_x] # [330, 750]
    ylim = [mid_y - half_stim_y, mid_y + half_stim_y] # [820, 1100]

    return xlim, ylim