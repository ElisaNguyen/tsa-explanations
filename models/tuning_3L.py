# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd
import torch
from CoreSNN import *

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)

dtype = torch.float

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

"""### Import data"""

dataset = load_obj('../data/dataset900.pkl')

X_train = dataset['X_train']
y_train = dataset['y_train']
X_val = dataset['X_val']
y_val = dataset['y_val']
X_test = dataset['X_test']
y_test = dataset['y_test']

"""## Hyperparameter tuning with greedy optimization

Meaning one parameter is tuned optimally at a time, we assume independence of hyperparameters
"""

nb_inputs = 14
nb_outputs = 11
max_time = 900
nb_steps = 900
nb_layers = 3

# hyperparams
time_steps = [1e-2, 1e-3, 1e-4]
nbs_hidden1 = [15, 30, 50]
nbs_hidden2 = [15, 30, 50]
tau_syns = [10e-4, 10e-3, 10e-2]
tau_mems = [10e-4, 10e-3, 10e-2]
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [128, 256, 512]


def train_for_params(time_step=1e-2, nb_hidden1=15, nb_hidden2=15, tau_syn=10e-3, tau_mem=10e-3, learning_rate=1e-3,
                     batch_size=256):
    """
    Run for greedy optimization
    """
    hyperparams = {'batch_size': batch_size,
                   'learning_rate': learning_rate,
                   'tau_mem': tau_mem,
                   'tau_syn': tau_syn,
                   'time_step': time_step,
                   'nb_hiddens': [nb_hidden1, nb_hidden2]}
    snn3L = SNN(hyperparams=hyperparams, nb_inputs=nb_inputs, nb_outputs=nb_outputs, nb_layers=nb_layers,
                        max_time=max_time)
    loss_hist_train = snn3L.train(X_train, y_train, path=path, early_stopping=False)
    valloss = snn3L.evaluate_loss(X_val, y_val)
    return valloss, loss_hist_train


def write_tuning_results(param, data, path):
    df = pd.DataFrame(data, columns=[param, "Validation loss", "Train loss history"])
    min_valloss = np.min(df["Validation loss"])
    best_p = df[df["Validation loss"] == min_valloss][param].to_numpy()[0]
    fname = path + "valloss_" + str(param) + ".csv"
    df.to_csv(fname, index=False)
    return best_p


# Greedy tuning, one parameter after the other
path = "../models/tuning/results_3L/"

# time_step
valloss_time_step = []
for time_step in time_steps:
    valloss, loss_hist_train = train_for_params(time_step=time_step)
    valloss_time_step.append([time_step, valloss, loss_hist_train])

best_time_step = write_tuning_results("Time step", valloss_time_step, path)

# nb_hidden1
valloss_nb_hidden1 = []
for nb_hidden1 in nbs_hidden1:
    valloss, loss_hist_train = train_for_params(time_step=best_time_step, nb_hidden1=nb_hidden1)
    valloss_nb_hidden1.append([nb_hidden1, valloss, loss_hist_train])

best_nb_hidden1 = write_tuning_results("Nb Hidden1", valloss_nb_hidden1, path)

# nb_hidden2
valloss_nb_hidden2 = []
for nb_hidden2 in nbs_hidden2:
    valloss, loss_hist_train = train_for_params(time_step=best_time_step, nb_hidden1=best_nb_hidden1,
                                                nb_hidden2=nb_hidden2)
    valloss_nb_hidden2.append([nb_hidden2, valloss, loss_hist_train])

best_nb_hidden2 = write_tuning_results("Nb Hidden2", valloss_nb_hidden2, path)

# tau syn
valloss_tau_syn = []
for tau_syn in tau_syns:
    valloss, loss_hist_train = train_for_params(time_step=best_time_step, nb_hidden1=best_nb_hidden1,
                                                nb_hidden2=best_nb_hidden2, tau_syn=tau_syn)
    valloss_tau_syn.append([tau_syn, valloss, loss_hist_train])

best_tau_syn = write_tuning_results("Tau syn", valloss_tau_syn, path)

# tau mem
valloss_tau_mem = []
for tau_mem in tau_mems:
    valloss, loss_hist_train = train_for_params(time_step=best_time_step, nb_hidden1=best_nb_hidden1,
                                                nb_hidden2=best_nb_hidden2, tau_syn=best_tau_syn, tau_mem=tau_mem)
    valloss_tau_mem.append([tau_mem, valloss, loss_hist_train])

best_tau_mem = write_tuning_results("Tau mem", valloss_tau_mem, path)

# learning rate
valloss_lr = []
for lr in learning_rates:
    valloss, loss_hist_train = train_for_params(time_step=best_time_step, nb_hidden1=best_nb_hidden1,
                                                nb_hidden2=best_nb_hidden2, tau_syn=best_tau_syn, tau_mem=best_tau_mem,
                                                learning_rate=lr)
    valloss_lr.append([lr, valloss, loss_hist_train])

best_lr = write_tuning_results("Learning rate", valloss_lr, path)

# batch size
valloss_bs = []
for bs in batch_sizes:
    valloss, loss_hist_train = train_for_params(time_step=best_time_step, nb_hidden1=best_nb_hidden1,
                                                nb_hidden2=best_nb_hidden2, tau_syn=best_tau_syn, tau_mem=best_tau_mem,
                                                learning_rate=best_lr, batch_size=bs)
    valloss_bs.append([bs, valloss, loss_hist_train])

best_bs = write_tuning_results("Batch size", valloss_bs, path)

best_params = {'time_step': best_time_step,
               'nb_hidden1': best_nb_hidden1,
               'nb_hidden2': best_nb_hidden2,
               'tau_syn': best_tau_syn,
               'tau_mem': best_tau_mem,
               'learning_rate': best_lr,
               'batch_size': best_bs}
save_obj(best_params, 'best_params_3L.pkl')
