# -*- coding: utf-8 -*-

import os
import random
import sys
import numpy as np
import torch
from tqdm import tqdm

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)

dtype = torch.float

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from ExplanationCreation import *
from ExplanationEvaluation import *

sys.path.insert(1, '../../models')
from CoreSNN import *


# Load data
dataset = load_obj('../data/dataset_max.pkl')

A_testset_t = load_obj('../data/quantitative_test_t_A.pkl')
B_testset_t = load_obj('../data/quantitative_test_t_B.pkl')
A_y_true = dataset['y_test_A'][:, A_testset_t]
B_y_true = dataset['y_test_B'][:, B_testset_t]

# Fixed parameters defined as global
nb_inputs = 14
nb_outputs = 11

"""# Get all the explanations for the quantitative analysis
so that it does not have to be recomputed for each metric
"""


def extract_explanations_for_quantitative_analysis(testset_t, nb_layers, X_data, y_data, explanation_type, filename):
    """
    Helper function to extract the X_spikes, explanations (attribution maps) and the prediction for a model
    :param explanation_type: string incidating the type of explanation
    :param testset_t: the timestamps to be run and extract explanations for
    :param nb_layers: amount of layers of the model
    :param X_data: data in the dictionary times, units form
    :param y_data: labels
    :param filename: string of the filename to save the information under
    """
    testset_explanations = {}
    for t in tqdm(testset_t):
        # get the relevant part of the dataset, this is done for performance reasons
        start_t = t - 3600 if t >= 3600 else 0
        X = {'times': X_data['times'][:, np.where((X_data['times'] >= start_t) & (X_data['times'] < t))[1]] - start_t,
             'units': X_data['units'][:, np.where((X_data['times'] >= start_t) & (X_data['times'] < t))[1]]}
        y = y_data[:, start_t:t]

        model = initiate_model(nb_layers, (t - start_t))

        # reset synaptic currents and membrane potentials to fit the data duration
        model.syns = []
        model.mems = []
        for l in range(model.nb_layers):
            model.syns.append(torch.zeros((len(y), model.layer_sizes[l + 1]), device=device, dtype=dtype))
            model.mems.append(torch.zeros((len(y), model.layer_sizes[l + 1]), device=device, dtype=dtype))

        y_pred, log_p_y, layer_recs = model.predict(X, y)
        probs = torch.exp(log_p_y)[0].t()
        data_generator = sparse_data_generator_from_spikes(X, y, len(y), model.layer_sizes[0],
                                                           model.max_time, shuffle=False)
        X_spikes, _ = next(data_generator)

        if explanation_type == 'sam':
            attribution = sam(model, X_spikes, layer_recs, probs[-1], t - start_t)
        else:
            attribution = tsa(model, X_spikes, layer_recs, probs[-1], t - start_t, explanation_type)
        prediction = y_pred[0][-1]
        e = attribution[prediction]

        testset_explanations[t] = (e.detach(), prediction)
        save_obj(testset_explanations, '../evaluation/' + filename + '.pkl')


expl_types = ['s', 'ns', 'sam']

for nb_layer in range(3):
    for expl_type in expl_types:
        # A
        extract_explanations_for_quantitative_analysis(A_testset_t, nb_layer, dataset['X_test_A'], dataset['y_test_A'],
                                                       expl_type, '{}/{}L_explanations_A'.format(expl_type, nb_layer))
        # B
        extract_explanations_for_quantitative_analysis(B_testset_t, nb_layer, dataset['X_test_B'], dataset['y_test_B'],
                                                       expl_type, '{}/{}L_explanations_B'.format(expl_type, nb_layer))