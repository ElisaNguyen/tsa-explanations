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

from ExplanationCreation import *
from ExplanationEvaluation import *

sys.path.insert(1, '../models')
from CoreSNN import *

# Load data
dataset = load_obj('../data/dataset_max.pkl')

"""# Continuity
Defined as max-sensitivity (maximum change in the attributions when perturbing the input)
"""


def is_contiguous(spike_data):
    """
    Function returning a boolean if spike_data is contiguous.
    :param spike_data:
    :return: Boolean
    """
    return np.all(np.diff(spike_data) == 1)


def perturb_duration(max_time, spike_data, perturbation, perturbation_loc, perturbation_size):
    """
    Function to perform the perturbation of activity duration
    :param max_time: maximum duration of the activities
    :param spike_data: activity spike data
    :param perturbation: string indicating whether to shorten or lengthen activity
    :param perturbation_loc: string indicating where the perturbation should be done
    :param perturbation_size: size by which original spike_data is perturbed
    :return: perturbed activity spike_data
    """
    if perturbation == 'shorten':
        if perturbation_loc == 'start':
            start = np.min(spike_data) + perturbation_size
            perturbed_data = np.clip(spike_data[np.where(spike_data >= start)], 0, max_time)
        elif perturbation_loc == 'end':
            end = np.max(spike_data) - perturbation_size
            perturbed_data = np.clip(spike_data[np.where(spike_data < end)], 0, max_time)
    elif perturbation == 'lengthen':
        if perturbation_loc == 'start':
            start = np.clip(np.min(spike_data) - perturbation_size, 0, np.min(spike_data))
            new_data = np.array(np.arange(start, np.min(spike_data)))
            perturbed_data = np.insert(spike_data, 0, new_data)
        elif perturbation_loc == 'end':
            end = np.clip(np.max(spike_data) + perturbation_size, np.max(spike_data), max_time)
            new_data = np.array(np.arange(np.max(spike_data) + 1, end))
            perturbed_data = np.insert(spike_data, -1, new_data)
    return perturbed_data


def draw_perturbation(sensor_spikes):
    """
    Function to randomly draw the perturbation variables that determine the perturbation
    :param sensor_spikes: Original activity spike_train of one sensor
    :return: Determined perturbation in terms of what kind, where and by how much
    """
    perturbation = np.random.choice(['shorten', 'lengthen'])
    perturbation_loc = np.random.choice(['start', 'end'])
    perturbation_size = np.random.choice(np.arange(np.round(0.1 * len(sensor_spikes)) + 1))
    return perturbation, perturbation_loc, perturbation_size


def perform_perturbation(max_time, sensor_spikes, sensor, X_times, X_units):
    """
    Function to perturb data for the evaluation of continuity
    :param max_time: maximum duration of perturbed data
    :param sensor_spikes: activity spike data
    :param sensor: active sensor
    :param X_times: indeces of firing times
    :param X_units: indeces of firing units
    :return: perturbed X_times and X_units
    """
    perturbation, perturbation_loc, perturbation_size = draw_perturbation(sensor_spikes)
    perturbed_sensor_data = perturb_duration(max_time, sensor_spikes, perturbation, perturbation_loc, perturbation_size)
    perturbed_sensor_data = [x for x in perturbed_sensor_data if x not in X_times]
    X_times.extend(perturbed_sensor_data)
    X_units.extend([sensor] * len(perturbed_sensor_data))
    return X_times, X_units


def perturb_sensor_dimension(X, sensor, max_time):
    """
    Function to apply continuity perturbation in a sensor dimension
    :param X: data
    :param sensor: activity sensor dimension
    :param max_time: maximum duration of a sample
    :return: perturbed sensor dimension in times, units format
    """
    X_perturbed_times = []
    X_perturbed_units = []
    sensor_spikes = X['times'][np.where(X['units'] == sensor)]
    if is_contiguous(sensor_spikes):
        X_perturbed_times, X_perturbed_units = perform_perturbation(max_time, sensor_spikes, sensor, X_perturbed_times,
                                                                    X_perturbed_units)
    else:
        edges = np.where(np.diff(sensor_spikes) != 1)[0] + 1
        start_edge = 0
        for edge in edges:
            X_perturbed_times, X_perturbed_units = perform_perturbation(max_time, sensor_spikes[start_edge:edge],
                                                                        sensor, X_perturbed_times, X_perturbed_units)
            start_edge = edge
        X_perturbed_times, X_perturbed_units = perform_perturbation(max_time, sensor_spikes[edges[-1]:], sensor,
                                                                    X_perturbed_times, X_perturbed_units)
    return X_perturbed_times, X_perturbed_units


def generate_perturbed_data(X_data, testset_t, savepath):
    """
    Function to generate perturbed data of X_data
    :param X_data: input data to be perturbed
    :param testset_t: timestamps of X_data
    :param savepath: path to save the perturbed data to
    :return:
    """
    perturbed_data = {}
    for t in tqdm(testset_t):
        # get the data
        start_t = t - 3600 if t > 3600 else 0
        X = {'times': X_data['times'][:, np.where((X_data['times'] >= start_t) & (X_data['times'] < t))[1]] - start_t,
             'units': X_data['units'][:, np.where((X_data['times'] >= start_t) & (X_data['times'] < t))[1]]}

        max_time = t - start_t

        # perturb the data
        X_perturbed = {'times': [], 'units': []}
        for sensor in np.unique(X['units']):
            X_sensor_perturbed_times, X_sensor_perturbed_units = perturb_sensor_dimension(X, sensor, max_time)
            X_perturbed['times'].extend(X_sensor_perturbed_times)
            X_perturbed['units'].extend(X_sensor_perturbed_units)

        # sort the data 
        sorted_X_p = np.transpose(np.array(sorted(zip(X_perturbed['times'], X_perturbed['units']))))

        # bring it back to the right shape
        X_perturbed['times'] = np.expand_dims(sorted_X_p[0], axis=0)
        X_perturbed['units'] = np.expand_dims(sorted_X_p[1], axis=0)

        perturbed_data[t] = X_perturbed
        save_obj(perturbed_data, savepath)
    return perturbed_data


def compute_perturbed_explanation(nb_layers, X_perturbed, y_data, testset_t, explanation_type, path):
    """
    Computes max sensitivity score as the maximum change in the explanation after perturbation.
    :param explanation_type: string indicating explanation type (s, ns, sam)
    :param nb_layers: number of layers of the SNN model
    :param X_perturbed: perturbed input data
    :param y_data: original labels
    :param testset_t: timestamps of X_perturbed
    :param path: path to save the explanations to
    :return: perturbed explanations in dictionary form
    """
    perturbed_explanations = {}
    for t in tqdm(testset_t):
        start_t = t - 3600 if t > 3600 else 0
        max_time = t - start_t
        y = y_data[:, start_t:t]

        # initiate the model
        model = initiate_model(nb_layers, max_time)

        # reset synaptic currents and membrane potentials to fit the data duration
        model.syns = []
        model.mems = []
        for l in range(model.nb_layers):
            model.syns.append(torch.zeros((len(y), model.layer_sizes[l + 1]), device=device, dtype=dtype))
            model.mems.append(torch.zeros((len(y), model.layer_sizes[l + 1]), device=device, dtype=dtype))

        # run the model on the data
        y_p_pred, log_p_y, layer_recs_p = model.predict(X_perturbed[t], y)
        probs_p = torch.exp(log_p_y)[0].t()

        # get the spiking data
        data_generator = sparse_data_generator_from_spikes(X_perturbed[t], y, 1, 14, max_time)
        X_p_spikes, _ = next(data_generator)

        if explanation_type == 'sam':
            attribution_p = ncs_attribution_map_mm(model, X_p_spikes, layer_recs_p, probs_p[-1], max_time, tsa_variant='s')
        else:
            attribution_p = attribution_map_mm(model, X_p_spikes, layer_recs_p, probs_p[-1], max_time, explanation_type)
        prediction_p = y_p_pred[0][-1]
        e_p = attribution_p[prediction_p]
        perturbed_explanations[t] = (e_p.detach(), prediction_p)
        save_obj(perturbed_explanations, path)
    return perturbed_explanations


def max_sensitivity_score(testset_t, model_explanations, perturbed_explanations):
    """
    Computes continuity as max-sensitivity score (max Frobenius norm of the difference).
    :param testset_t: timestamp (i.e. sample indeces)
    :param model_explanations: original explanations of the model behavior on clean data
    :param perturbed_explanations: explanations of the model behavior on perturbed data
    :return: score
    """
    sensitivities = []
    for t in tqdm(testset_t):
        e, _ = model_explanations[t]
        e_p, _ = perturbed_explanations[t]
        sensitivity = torch.norm(e - e_p).cpu().numpy()
        sensitivities.append(sensitivity)
    max_sensitivity = max(sensitivities)
    return max_sensitivity


A_testset_t = load_obj('../data/quantitative_test_t_A.pkl')
B_testset_t = load_obj('../data/quantitative_test_t_B.pkl')
A_y_true = dataset['y_test_A'][:, A_testset_t]
B_y_true = dataset['y_test_B'][:, B_testset_t]
expl_types = ['s', 'ns', 'sam']

with torch.no_grad():
    # generate the perturbed data
    X_perturbed_A = generate_perturbed_data(dataset['X_test_A'], A_testset_t,
                                            '../evaluation/continuity/perturbed_X_A.pkl')
    X_perturbed_B = generate_perturbed_data(dataset['X_test_B'], B_testset_t,
                                            '../evaluation/continuity/perturbed_X_B.pkl')

    # TSA-S
    for nb_layer in range(3):
        for expl_type in expl_types:
            print('Evaluating continuity for {} explanations of {}L-SNN...'.format(expl_type, nb_layer))
            # A
            perturbed_explanations = compute_perturbed_explanation(nb_layer, X_perturbed_A, dataset['y_test_A'],
                                                                   A_testset_t, expl_type,
                                                                   '../evaluation/continuity/{}/perturbed_explanations_{}A.pkl'.format(
                                                                       expl_type, nb_layer))
            model_explanations = load_obj('../evaluation/{}/{}L_explanations_A.pkl'.format(expl_type, nb_layer))
            max_sensitivity = max_sensitivity_score(A_testset_t, model_explanations, perturbed_explanations)
            save_obj(max_sensitivity, '../evaluation/sensitivity/{}/max_sensitivity_{}A.pkl'.format(expl_type, nb_layer))
            # B
            perturbed_explanations = compute_perturbed_explanation(nb_layer, X_perturbed_B, dataset['y_test_B'],
                                                                   B_testset_t, expl_type,
                                                                   '../evaluation/continuity/{}/perturbed_explanations_{}B.pkl'.format(
                                                                       expl_type, nb_layer
                                                                   ))
            model_explanations = load_obj('../evaluation/{}/{}L_explanations_B.pkl'.format(expl_type, nb_layer))
            max_sensitivity = max_sensitivity_score(B_testset_t, model_explanations, perturbed_explanations)
            save_obj(max_sensitivity, '../evaluation/continuity/{}/max_sensitivity_{}B.pkl'.format(expl_type, nb_layer))
        print('Continuity evaluation of SNN-{}L is done!'.format(nb_layer))
