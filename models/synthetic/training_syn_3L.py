# -*- coding: utf-8 -*-

import random
import sys
import numpy as np
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

sys.path.insert(1, '../')
from CoreSNN import *

"""### Import data"""

dataset = load_obj('../../data/synthetic/syn_data900.pkl')

X_train = dataset['X_train']
y_train = dataset['y_train']
X_test = dataset['X_test']
y_test = dataset['y_test']

"""### Setup of the spiking network model"""

hyperparams = {'time_step': 0.001,
               'nb_hidden1': 10,
               'nb_hidden2': 10,
               'tau_syn': 0.01,
               'tau_mem': 0.00001,
               'optimizer': optim.Adam,
               'learning_rate': 0.01,
               'batch_size': 128}

hyperparams['nb_hiddens'] = [hyperparams['nb_hidden1'], hyperparams['nb_hidden2']]

nb_inputs = 3
nb_outputs = 4
nb_layers = 3
max_time = 900
nb_steps = 900 

ThreeLayerSNN = SNN(hyperparams=hyperparams, 
                  nb_inputs=nb_inputs, 
                  nb_outputs=nb_outputs, 
                  nb_layers=nb_layers, 
                  nb_steps=nb_steps, 
                  max_time=max_time)

"""## Training the network"""

model_save_path = 'training/results_3L/'
loss_hist = ThreeLayerSNN.train(X_train, y_train, path=model_save_path)
save_obj(loss_hist, model_save_path+"loss_hist_3L.pkl")

