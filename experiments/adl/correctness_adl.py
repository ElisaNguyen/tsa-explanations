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

sys.path.insert(1, '../')
from ExplanationEvaluation import *
sys.path.insert(1, '../../models')
from CoreSNN import *

# Load data
dataset = load_obj('../../data/adl/dataset_max.pkl')

# Fixed parameters
nb_inputs = 14
nb_outputs = 11


expl_types = ['s', 'ns', 'sam']
with torch.no_grad():
    A_testset_t = load_obj('../../data/adl/quantitative_test_t_A.pkl')
    B_testset_t = load_obj('../../data/adl/quantitative_test_t_B.pkl')
    A_y_true = dataset['y_test_A'][:, A_testset_t]
    B_y_true = dataset['y_test_B'][:, B_testset_t]

    for nb_layer in range(3):
        for expl_type in expl_types:
            print('Evaluating correctness for {} explanations of {}L-SNN...'.format(expl_type, nb_layer))
            # A
            model_explanations = load_obj('explanations/{}/{}L_explanations_A.pkl'.format(expl_type, nb_layer))
            y_preds_perturbed, y_preds_clean = flip_and_predict(nb_layer, dataset['X_test_A'], dataset['y_test_A'],
                                                                model_explanations,
                                                                A_testset_t)
            save_obj(y_preds_perturbed,
                     'correctness/{}/y_preds_perturbed_{}L_A.pkl'.format(expl_type, nb_layer))
            # B
            model_explanations = load_obj('explanations/{}/{}L_explanations_B.pkl'.format(expl_type, nb_layer))
            y_preds_perturbed, y_preds_clean = flip_and_predict(nb_layer, dataset['X_test_B'], dataset['y_test_B'],
                                                                model_explanations,
                                                                B_testset_t)
            save_obj(y_preds_perturbed,
                     'correctness/{}/y_preds_perturbed_{}L_B.pkl'.format(expl_type, nb_layer))
        print('Correctness evaluation of SNN-{}L is done!'.format(nb_layer))
