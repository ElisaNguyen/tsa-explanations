# -*- coding: utf-8 -*-
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

from ExplanationEvaluation import *
sys.path.insert(1, '../../models')
from CoreSNN import *

# Load data
dataset = load_obj('../data/dataset_max.pkl')

# Fixed parameters
nb_inputs = 14
nb_outputs = 11

"""# Correctness

Correctness is measured in explanation selectivity. 
This is the average AUC of the graphs resulting from flipping the most important feature segments of the explanation.
"""


def flip_segment(X_spikes, segment):
    """
    Flips the values of a segment in X_spikes format
    :param X_spikes: spiking input data from spike generator
    :param segment: segment in X_spikes to be flipped
    :return: spiking data with flipped segment
    """
    _, (d, t_start, t_end) = segment
    X_perturbed = X_spikes.to_dense()
    X_perturbed[:, t_start:t_end, d] = torch.abs(X_perturbed[:, t_start:t_end, d] - 1)
    X_perturbed = X_perturbed.to_sparse()
    return X_perturbed


def flip_and_predict(nb_layers, X_data, y_data, model_explanations, testset_t):
    """
    Function to get the predictions of the model with nb_layers on X_data when flipping the feature segments
    :param nb_layers: number of layers of the SNN model
    :param X_data: input data
    :param y_data: labels
    :param model_explanations: extracted explanations for X_data
    :param testset_t: timestamps that are part of the testset
    :return: model predictions for perturbed data and original predictions
    """
    # define the model for the specific duration (performs same though because the weights are the same)
    model = initiate_model(nb_layers, 1)
    # for storing the predictions with flipped segments
    # Will be an array of duration arrays of each the length of how many segments they have.
    y_preds_flipped = []
    y_preds = []
    # go through the "samples"
    for t in tqdm(testset_t):
        e, prediction = model_explanations[t]
        y_preds.append(prediction)
        start_t = t - 3600 if t >= 3600 else 0

        model.nb_steps = t - start_t
        model.max_time = t - start_t

        # get the relevant part of the dataset, this is done for performance reasons
        X = {'times': X_data['times'][:, np.where((X_data['times'] >= start_t) & (X_data['times'] < t))[1]] - start_t,
             'units': X_data['units'][:, np.where((X_data['times'] >= start_t) & (X_data['times'] < t))[1]]}
        y = y_data[:, start_t:t]
        data_generator = sparse_data_generator_from_spikes(X, y, len(y), model.layer_sizes[0],
                                                           model.max_time, shuffle=False)
        X_spikes, _ = next(data_generator)

        # idenfity feature segments in e that are positively or negatively attributing
        feature_segments = segment_features(e)

        # rank the segments
        ranked_fs = rank_segments(e, feature_segments)

        y_pred_perturbed = []
        X_perturbed = X_spikes
        for i, segment in enumerate(ranked_fs):
            X_perturbed = flip_segment(X_perturbed, segment)

            # Evaluate & record y_pred for the perturbed input
            pred_perturbed, _, _ = predict_from_spikes(model, X_perturbed)
            y_pred_perturbed.append(pred_perturbed[0][-1])
        y_preds_flipped.append(y_pred_perturbed)
    return y_preds_flipped, y_preds


expl_types = ['s', 'ns', 'sam']
with torch.no_grad():
    A_testset_t = load_obj('../data/quantitative_test_t_A.pkl')
    B_testset_t = load_obj('../data/quantitative_test_t_B.pkl')
    A_y_true = dataset['y_test_A'][:, A_testset_t]
    B_y_true = dataset['y_test_B'][:, B_testset_t]

    for nb_layer in range(3):
        for expl_type in expl_types:
            print('Evaluating correctness for {} explanations of {}L-SNN...'.format(expl_type, nb_layer))
            # A
            model_explanations = load_obj('../evaluation/{}/{}L_explanations_A.pkl'.format(expl_type, nb_layer))
            y_preds_perturbed, y_preds_clean = flip_and_predict(nb_layer, dataset['X_test_A'], dataset['y_test_A'],
                                                                model_explanations,
                                                                A_testset_t)
            save_obj(y_preds_perturbed,
                     '../evaluation/correctness/{}/y_preds_perturbed_{}L_A.pkl'.format(expl_type, nb_layer))
            # B
            model_explanations = load_obj('../evaluation/{}/{}L_explanations_B.pkl'.format(expl_type, nb_layer))
            y_preds_perturbed, y_preds_clean = flip_and_predict(nb_layer, dataset['X_test_B'], dataset['y_test_B'],
                                                                model_explanations,
                                                                B_testset_t)
            save_obj(y_preds_perturbed,
                     '../evaluation/correctness/{}/y_preds_perturbed_{}L_B.pkl'.format(expl_type, nb_layer))
        print('Correctness evaluation of SNN-{}L is done!'.format(nb_layer))
