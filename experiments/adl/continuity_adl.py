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
from ExplanationCreation import *
from ExplanationEvaluation import *

sys.path.insert(1, '../../models')
from CoreSNN import *

# Load data
dataset = load_obj('../../data/adl/dataset_max.pkl')

"""# Continuity
Defined as max-sensitivity (maximum change in the attributions when perturbing the input)
"""

A_testset_t = load_obj('../../data/adl/quantitative_test_t_A.pkl')
B_testset_t = load_obj('../../data/adl/quantitative_test_t_B.pkl')
A_y_true = dataset['y_test_A'][:, A_testset_t]
B_y_true = dataset['y_test_B'][:, B_testset_t]
expl_types = ['s', 'ns', 'sam']

with torch.no_grad():
    # generate the perturbed data
    X_perturbed_A = generate_perturbed_data(dataset['X_test_A'], A_testset_t,
                                            'continuity/perturbed_X_A.pkl')
    X_perturbed_B = generate_perturbed_data(dataset['X_test_B'], B_testset_t,
                                            'continuity/perturbed_X_B.pkl')

    for nb_layer in range(3):
        for expl_type in expl_types:
            print('Evaluating continuity for {} explanations of {}L-SNN...'.format(expl_type, nb_layer))
            # A
            perturbed_explanations = compute_perturbed_explanation(nb_layer, X_perturbed_A, dataset['y_test_A'],
                                                                   A_testset_t, expl_type,
                                                                   '.continuity/{}/perturbed_explanations_{}A.pkl'.format(
                                                                       expl_type, nb_layer))
            model_explanations = load_obj('explanations/{}/{}L_explanations_A.pkl'.format(expl_type, nb_layer))
            max_sensitivity = max_sensitivity_score(A_testset_t, model_explanations, perturbed_explanations)
            save_obj(max_sensitivity, 'continuity/{}/max_sensitivity_{}A.pkl'.format(expl_type, nb_layer))
            # B
            perturbed_explanations = compute_perturbed_explanation(nb_layer, X_perturbed_B, dataset['y_test_B'],
                                                                   B_testset_t, expl_type,
                                                                   'continuity/{}/perturbed_explanations_{}B.pkl'.format(
                                                                       expl_type, nb_layer
                                                                   ))
            model_explanations = load_obj('explanations/{}/{}L_explanations_B.pkl'.format(expl_type, nb_layer))
            max_sensitivity = max_sensitivity_score(B_testset_t, model_explanations, perturbed_explanations)
            save_obj(max_sensitivity, 'continuity/{}/max_sensitivity_{}B.pkl'.format(expl_type, nb_layer))
        print('Continuity evaluation of SNN-{}L is done!'.format(nb_layer))
