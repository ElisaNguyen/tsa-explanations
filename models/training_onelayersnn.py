# -*- coding: utf-8 -*-

import os
import random
import sys

import numpy as np
import pandas as pd

import torch


random.seed(123)
torch.manual_seed(123)
np.random.seed(123)

dtype = torch.float

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")

print(device)

sys.path.insert(1, '/local/work/enguyen/tuning')
from CoreSNN import *

"""### Import data and transform it to right format"""

dataset = load_obj('/local/work/enguyen/data/dataset900.pkl')

X_train = dataset['X_train']
y_train = dataset['y_train']
X_val = dataset['X_val']
y_val = dataset['y_val']
X_test = dataset['X_test']
y_test = dataset['y_test']

"""### Setup of the spiking network model"""

hyperparams = load_obj('/local/work/enguyen/tuning/results_onelayersnn/best_params.pkl')
print(hyperparams)

nb_inputs = 14
nb_outputs = 11
nb_layers = 1
max_time = 900

OneLayerSNN = SNN(hyperparams=hyperparams, 
                  nb_inputs=nb_inputs, 
                  nb_outputs=nb_outputs, 
                  nb_layers=nb_layers,
                  max_time=max_time)

"""## Training the network"""

model_save_path = '/local/work/enguyen/training/onelayersnn/'
loss_hist = OneLayerSNN.train(X_train, y_train, path=model_save_path)
save_obj(loss_hist, model_save_path+"loss_hist.pkl")