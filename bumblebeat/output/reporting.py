import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bumblebeat.utils.data as utils

path = 'gpu_run-groove/full-midionly/20200630-191931/log.txt'


def extract_timestep(l):
    """
    Extract timestep from line, <l>
    """
    return int(re.findall('step\s*\d*',l)[0].split(' ')[-1])


def extract_train_loss(l):
    """
    Extract train_loss from line, <l>
    """
    return float(re.findall('loss\s*\d*.\d*',l)[0].split(' ')[-1])


def extract_valid_loss(l):
    """
    Extract valid_loss from line, <l>
    """
    return float(re.findall('valid loss\s*\d*.\d*',l)[0].split(' ')[-1])


def parse_log(path):
    """
    Parse model log at <path> to dict of useful stats
    """
    with open(path, 'r') as f:
        log = f.read()

    log_d = {
        'time_step':[],
        'train_loss':[],
        'valid_loss':[]
    }
    for l in log.split('\n'):
        if '| Eval' in l:
            log_d['time_step'].append(extract_timestep(l))
            log_d['train_loss'].append(np.nan)
            log_d['valid_loss'].append(extract_valid_loss(l))
        elif '| epoch' in l:
            log_d['time_step'].append(extract_timestep(l))
            log_d['train_loss'].append(extract_train_loss(l))
            log_d['valid_loss'].append(np.nan)

    return pd.DataFrame(log_d)


def loss_plot(in_path, out_path):
    """
    Generate plot of train and valid loss from log
    at <in_path>. Store .png at <out_path>
    """
    df = parse_log(in_path)

    fig = plt.figure(figsize=(8,5))
    plt.title(in_path)
    plt.xlabel('Time Step')
    plt.ylabel('Loss')
    plt.grid(which='both')

    train = df[~df['train_loss'].isnull()]
    plt.plot(train['time_step'].values, train['train_loss'].values, color='green', label='Train')

    valid = df[~df['valid_loss'].isnull()]
    plt.plot(valid['time_step'].values, valid['valid_loss'].values, color='orange', label='Valid')

    plt.legend(loc='upper right')
    utils.create_dir_if_not_exists(out_path)
    plt.savefig(out_path)