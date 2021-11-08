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

print(device)

sys.path.insert(1, '/local/work/enguyen')
from CoreSNN import *
from ExplanationCreation import *
from ExplanationEvaluation import *

# Load data
dataset = load_obj('/local/work/enguyen/data/dataset_max.pkl')

A_testset_t = load_obj('/local/work/enguyen/data/quantitative_test_t_A_final.pkl')
B_testset_t = load_obj('/local/work/enguyen/data/quantitative_test_t_B_final.pkl')

A_y_true = dataset['y_test_A'][:, A_testset_t]
B_y_true = dataset['y_test_B'][:, B_testset_t]

# Fixed parameters defined as global
nb_inputs = 14
nb_outputs = 11

"""# Get all the explanations for the quantitative analysis
so that it does not have to be recomputed for each metric
"""


def extract_information_for_quantitative_analysis(testset_t, nb_layers, X_data, y_data, variant, filename):
    """
    Helper function to extract the X_spikes, explanations (attribution maps) and the prediction for a model
    :param variant: TSA variant (string, 's' or 'ns')
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

        attribution = attribution_map_mm(model, X_spikes, layer_recs, probs[-1], t - start_t, variant)
        prediction = y_pred[0][-1]
        e = attribution[prediction]

        testset_explanations[t] = (e.detach(), prediction)
        save_obj(testset_explanations, '/local/work/enguyen/evaluation/' + filename + '.pkl')


def assign_random_attribution(X_spikes, min_attr, max_attr):
    random_e = torch.zeros(X_spikes.shape).to(device)
    random_e[X_spikes != 0] = torch.Tensor(np.random.uniform(min_attr, max_attr, X_spikes[X_spikes != 0].shape)).to(
        device)
    return X_spikes


def generate_baseline_data(testset_t, X_data, y_data, path):
    baseline_explanations = {}
    max_attr = -50
    min_attr = 50
    for filename in os.listdir('/local/work/enguyen/evaluation/tsa-s'):
        f = os.path.join('/local/work/enguyen/evaluation/tsa-s', filename)
        if os.path.isfile(f):
            max_attr = max(max_attr, get_max_attr(f))
            min_attr = min(min_attr, get_min_attr(f))
    for filename in os.listdir('/local/work/enguyen/evaluation/tsa-ns'):
        f = os.path.join('/local/work/enguyen/evaluation/tsa-ns', filename)
        if os.path.isfile(f):
            max_attr = max(max_attr, get_max_attr(f))
            min_attr = min(min_attr, get_min_attr(f))

    for t in tqdm(testset_t):
        # get the relevant part of the dataset, this is done for performance reasons
        start_t = t - 3600 if t >= 3600 else 0
        X = {'times': X_data['times'][:, np.where((X_data['times'] >= start_t) & (X_data['times'] < t))[1]] - start_t,
             'units': X_data['units'][:, np.where((X_data['times'] >= start_t) & (X_data['times'] < t))[1]]}
        y = y_data[:, start_t:t]
        max_time = t - start_t

        data_generator = sparse_data_generator_from_spikes(X, y, len(y), 14, max_time, shuffle=False)
        X_spikes, _ = next(data_generator)

        baseline_explanations[t] = (
            assign_random_attribution(X_spikes.to_dense()[0].t(), min_attr, max_attr).detach(), y[0][-1])
        save_obj(baseline_explanations, path)


# TSA-S explanations
extract_information_for_quantitative_analysis(A_testset_t, 1, dataset['X_test_A'], dataset['y_test_A'], 's',
                                              'tsa-s/onelayer_explanations_A')
extract_information_for_quantitative_analysis(B_testset_t, 1, dataset['X_test_B'], dataset['y_test_B'], 's',
                                              'tsa-s/onelayer_explanations_B')
extract_information_for_quantitative_analysis(A_testset_t, 2, dataset['X_test_A'], dataset['y_test_A'], 's',
                                              'tsa-s/twolayer_explanations_A')
extract_information_for_quantitative_analysis(B_testset_t, 2, dataset['X_test_B'], dataset['y_test_B'], 's',
                                              'tsa-s/twolayer_explanations_B')
extract_information_for_quantitative_analysis(A_testset_t, 3, dataset['X_test_A'], dataset['y_test_A'], 's',
                                              'tsa-s/threelayer_explanations_A')
extract_information_for_quantitative_analysis(B_testset_t, 3, dataset['X_test_B'], dataset['y_test_B'], 's',
                                              'tsa-s/threelayer_explanations_B')

# TSA-NS explanations
extract_information_for_quantitative_analysis(A_testset_t, 1, dataset['X_test_A'], dataset['y_test_A'], 'ns',
                                              'tsa-ns/onelayer_explanations_A')
extract_information_for_quantitative_analysis(B_testset_t, 1, dataset['X_test_B'], dataset['y_test_B'], 'ns',
                                              'tsa-ns/onelayer_explanations_B')
extract_information_for_quantitative_analysis(A_testset_t, 2, dataset['X_test_A'], dataset['y_test_A'], 'ns',
                                              'tsa-ns/twolayer_explanations_A')
extract_information_for_quantitative_analysis(B_testset_t, 2, dataset['X_test_B'], dataset['y_test_B'], 'ns',
                                              'tsa-ns/twolayer_explanations_B')
extract_information_for_quantitative_analysis(A_testset_t, 3, dataset['X_test_A'], dataset['y_test_A'], 'ns',
                                              'tsa-ns/threelayer_explanations_A')
extract_information_for_quantitative_analysis(B_testset_t, 3, dataset['X_test_B'], dataset['y_test_B'], 'ns',
                                              'tsa-ns/threelayer_explanations_B')

generate_baseline_data(A_testset_t, dataset['X_test_A'], dataset['y_test_A'],
                       '/local/work/enguyen/evaluation/baseline_explanations_A.pkl')
generate_baseline_data(B_testset_t, dataset['X_test_B'], dataset['y_test_B'],
                       '/local/work/enguyen/evaluation/baseline_explanations_B.pkl')
