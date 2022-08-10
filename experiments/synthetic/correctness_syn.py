import os
import random
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import metrics
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

sys.path.insert(1, '../../models')
from CoreSNN import *

sys.path.insert(1, '../')
from ExplanationCreation import *
from ExplanationEvaluation import *

# Load data
dataset = load_obj('../../data/synthetic/syn_data.pkl')
nb_inputs = 3
nb_outputs = 4

expl_types = ['s', 'ns', 'sam']
with torch.no_grad():
    testset_t = load_obj('../../data/synthetic/expl_syn_testset.pkl')
    y_true = dataset['y_test'][:, testset_t]

    for nb_layer in range(3):
        for expl_type in expl_types:
            print('Evaluating correctness for {} explanations of {}L-SNN...'.format(expl_type, nb_layer))
            model_explanations = load_obj('explanations/{}/{}L_explanations.pkl'.format(expl_type, nb_layer))
            y_preds_perturbed, y_preds_clean = flip_and_predict(nb_layer, dataset['X_test'], dataset['y_test'],
                                                                model_explanations,
                                                                testset_t)
            save_obj(y_preds_perturbed,
                     'correctness/{}/y_preds_perturbed_{}L.pkl'.format(expl_type, nb_layer))
        print('Correctness evaluation of SNN-{}L is done!'.format(nb_layer))
   
