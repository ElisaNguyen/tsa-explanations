# -*- coding: utf-8 -*-
"""QuantitativeEvaluation - Playground.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kEoqH20gWonAYJ1YBCgkdKuYRq96FUW3

# Import libraries, models, data
"""

import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import matplotlib.pyplot as plt
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
from ExplanationCreationGeneral import *
from ExplanationEvaluationNEW import *

# Load data
dataset = load_obj('/local/work/enguyen/data/syn_data.pkl')

"""# Sensitivity

Defined as max-sensitivity (maximum change in the attributions when perturbing the input)

## Natural perturbations

Natural perturbations do not include broken sensors as this was not present in the training data and therefore is completely out of distribution. 

Therefore, domain knowledge about activities of daily living are used to manually manipulate the data. Here, there are the following possible natural perturbations.

* Activities are slightly longer/shorter --> one of each class will be randomly perturbed this way with a change of a minute/some percentage of the activity? 
* Activities are switched where it makes sense (e.g. showering before using bathroom) --> examine the examples and then decide, one of each class as well (switch of activities in the data leading up to the time step) --> but then this is also changing the ground truth labels, and rather checking sensitivity of the model
* Activities are squeezed into others (going to the bathroom during the night/while relaxing) --> applied to sleeping, spare timebut then this is also changing the ground truth labels, and rather checking sensitivity of the model


What does not make sense: 
* After leaving, the person does things in the house
* Switching of meal activities - these are connected to the time of day

**Therefore, only making it slightly longer and shorter**
"""

def initiate_model(nb_layers, t):
    """
    Function that initiates a SNN model with nb_layers which runs data of duration t, only defined for 3 layers
    :param nb_layers: (int) to define the number of layers
    :param t: max_time and nb_steps is defined by this (int)
    """
    nb_inputs = 3
    nb_outputs = 4
    if nb_layers == 1:
        params_onelayer = {'time_step': 0.001,
               'tau_syn': 0.01,
               'tau_mem': 0.001,
               'optimizer': optim.Adam,
               'learning_rate': 0.01,
               'batch_size': 128}
        model = SNN(hyperparams=params_onelayer, 
                          nb_inputs=nb_inputs, 
                          nb_outputs=nb_outputs, 
                          nb_layers=1, 
                          nb_steps=t, 
                          max_time=t)

        model.inference('/local/work/enguyen/synthetic/one_weights.pt')
        return model
    elif nb_layers == 2:
        params_twolayer = {'time_step': 0.001,
               'tau_syn': 0.01,
               'tau_mem': 0.001,
               'optimizer': optim.Adam,
               'learning_rate': 0.01,
               'batch_size': 128,
               'nb_hiddens': [10]}
        model = SNN(hyperparams=params_twolayer, 
                          nb_inputs=nb_inputs, 
                          nb_outputs=nb_outputs, 
                          nb_layers=2, 
                          nb_steps=t, 
                          max_time=t)

        model.inference('/local/work/enguyen/synthetic/two_weights.pt')
        return model
    elif nb_layers == 3:
        params_threelayer = {'time_step': 0.001,
               'tau_syn': 0.01,
               'tau_mem': 0.001,
               'optimizer': optim.Adam,
               'learning_rate': 0.01,
               'batch_size': 128,
               'nb_hiddens': [10, 10]}
        model = SNN(hyperparams=params_threelayer, 
                          nb_inputs=nb_inputs, 
                          nb_outputs=nb_outputs, 
                          nb_layers=3, 
                          nb_steps=t, 
                          max_time=t)
        model.inference('/local/work/enguyen/synthetic/three_weights.pt')
        return model


def is_contiguous(spike_data):
    return np.all(np.diff(spike_data) == 1)


def natural_perturbation(max_time, spike_data, perturbation, perturbation_loc, perturbation_size):
    if perturbation == 'shorten':
        if perturbation_loc == 'start': 
            start = np.min(spike_data)+perturbation_size
            perturbed_data = np.clip(spike_data[np.where(spike_data>=start)], 0, max_time)
        elif perturbation_loc == 'end':
            end = np.max(spike_data) - perturbation_size        
            perturbed_data = np.clip(spike_data[np.where(spike_data<end)], 0, max_time)
    elif perturbation == 'lengthen':
        if perturbation_loc == 'start':
            start = np.clip(np.min(spike_data)-perturbation_size, 0, np.min(spike_data))
            new_data = np.array(np.arange(start, np.min(spike_data)))
            perturbed_data = np.insert(spike_data, 0, new_data)
        elif perturbation_loc == 'end':
            end = np.clip(np.max(spike_data)+perturbation_size, np.max(spike_data), max_time)
            new_data = np.array(np.arange(np.max(spike_data)+1, end))
            perturbed_data = np.insert(spike_data, -1, new_data)
    return perturbed_data


def draw_perturbation(sensor_spikes):
    perturbation = np.random.choice(['shorten', 'lengthen'])
    perturbation_loc = np.random.choice(['start', 'end'])
    perturbation_size = np.random.choice(np.arange(np.round(0.1*len(sensor_spikes))+1))
    return perturbation, perturbation_loc, perturbation_size


def perform_perturbation(max_time, sensor_spikes, sensor, X_perturbed_times, X_perturbed_units):
    perturbation, perturbation_loc, perturbation_size = draw_perturbation(sensor_spikes)
    perturbed_sensor_data = natural_perturbation(max_time, sensor_spikes, perturbation, perturbation_loc, perturbation_size)
    perturbed_sensor_data = [x for x in perturbed_sensor_data if x not in X_perturbed_times]
    X_perturbed_times.extend(perturbed_sensor_data)
    X_perturbed_units.extend([sensor]*len(perturbed_sensor_data))
    return X_perturbed_times, X_perturbed_units


def perturb_sensor_dimension(X, sensor, max_time):
    X_perturbed_times = []
    X_perturbed_units = []
    sensor_spikes = X['times'][np.where(X['units'] == sensor)]
    if is_contiguous(sensor_spikes):
        X_perturbed_times, X_perturbed_units = perform_perturbation(max_time, sensor_spikes, sensor, X_perturbed_times, X_perturbed_units)
    else: 
        edges = np.where(np.diff(sensor_spikes)!=1)[0]+1
        start_edge = 0
        for edge in edges:
            X_perturbed_times, X_perturbed_units = perform_perturbation(max_time, sensor_spikes[start_edge:edge], sensor, X_perturbed_times, X_perturbed_units) 
            start_edge = edge
        X_perturbed_times, X_perturbed_units = perform_perturbation(max_time, sensor_spikes[edges[-1]:], sensor, X_perturbed_times, X_perturbed_units)
    return X_perturbed_times, X_perturbed_units


def generate_perturbed_data(X_data, testset_t, savepath):
    perturbed_data = {}
    for t in tqdm(testset_t): 
        # get the data
        start_t = t-1000 if t>1000 else 0 
        X = {'times': X_data['times'][:, np.where((X_data['times']>=start_t) & (X_data['times']<t))[1]]-start_t, 
              'units': X_data['units'][:, np.where((X_data['times']>=start_t) & (X_data['times']<t))[1]]}

        max_time = t-start_t
        
        # perturb the data
        X_perturbed = {'times': [], 'units': []}
        for sensor in np.unique(X['units']):
            X_sensor_perturbed_times, X_sensor_perturbed_units = perturb_sensor_dimension(X, sensor, max_time)
            X_perturbed['times'].extend(X_sensor_perturbed_times)
            X_perturbed['units'].extend(X_sensor_perturbed_units)
        
        # sort the data 
        sorted_X_p = np.transpose(np.array(sorted(zip(X_perturbed['times'], X_perturbed['units']))))

        # bring it back to the right shape
        X_perturbed['times'] = np.expand_dims(sorted_X_p[0], axis = 0)
        X_perturbed['units'] = np.expand_dims(sorted_X_p[1], axis = 0)
        
        perturbed_data[t] = X_perturbed
        save_obj(perturbed_data, savepath)
    return perturbed_data


def compute_perturbed_explanation(nb_layers, X_perturbed, y_data, testset_t, path, explanation_type):
    """
    Computes max sensitivity score as the maximum change in the explanation after perturbation. 
    """
    perturbed_explanations = {}
    for t in tqdm(testset_t):
        start_t = t-1000 if t>1000 else 0 
        max_time = t-start_t
        y = y_data[:, start_t:t]
        
        # initiate the model
        model = initiate_model(nb_layers, max_time)
        
        #reset synaptic currents and membrane potentials to fit the data duration 
        model.syns=[]
        model.mems=[]
        for l in range(model.nb_layers):
            model.syns.append(torch.zeros((len(y),model.layer_sizes[l+1]), device=device, dtype=dtype))
            model.mems.append(torch.zeros((len(y),model.layer_sizes[l+1]), device=device, dtype=dtype))
        
        # run the model on the data
        y_p_pred, log_p_y, layer_recs_p = model.predict(X_perturbed[t], y)
        probs_p = torch.exp(log_p_y)[0].t()

        #get the spiking data
        data_generator = sparse_data_generator_from_spikes(X_perturbed[t], y, 1, max_time, 14, max_time)
        X_p_spikes, _ = next(data_generator)
        
        if explanation_type == 'ncs':
            attribution_p = ncs_attribution_map_mm(model, X_p_spikes, layer_recs_p, probs_p[-1], max_time, tsa_variant='s')
        else: 
            attribution_p = attribution_map_mm(model, X_p_spikes, layer_recs_p, probs_p[-1], max_time, explanation_type)
            
        prediction_p = y_p_pred[0][-1]
        e_p = attribution_p[prediction_p]
        perturbed_explanations[t] = (e_p.detach(), prediction_p)
        save_obj(perturbed_explanations, path)
    return perturbed_explanations
    

def max_sensitivity_score(testset_t, model_explanations, perturbed_explanations): 
    sensitivities = []
    for t in tqdm(testset_t):
        e, _ = model_explanations[t]
        e_p, _ = perturbed_explanations[t]
        sensitivity = torch.norm(e-e_p).cpu().numpy()
        sensitivities.append(sensitivity)
    max_sensitivity = max(sensitivities)
    return max_sensitivity



testset_t = load_obj('/local/work/enguyen/data/expl_syn_testset.pkl')
y_true = dataset['y_test'][:, testset_t]

expl_path = '/local/work/enguyen/nocw/'
expl_types = ['ns2', 's']

for expl_type in expl_types:
    with torch.no_grad():
        # generate the perturbed data
        X_perturbed = generate_perturbed_data(dataset['X_test'], testset_t, '/local/work/enguyen/nocw/continuity/syn/naturally_perturbed_X.pkl')

        # ONLY LOAD THE DATA SINCE THE EVALUATION SHOULD BE ON THE SAME PERTURBED DATA, only do if needed
#         X_perturbed = load_obj('/local/work/enguyen/evaluation/sensitivity/naturally_perturbed_X.pkl')

        # OneLayerSNN
        
        perturbed_explanations = compute_perturbed_explanation(1, X_perturbed, dataset['y_test'], testset_t, expl_path+'continuity/syn/'+expl_type+'/perturbed_explanations_one.pkl', expl_type)
        model_explanations = load_obj(expl_path+'expl_one_syn_nocw_'+expl_type+'.pkl')

        max_sensitivity = max_sensitivity_score(testset_t, model_explanations, perturbed_explanations)
        save_obj(max_sensitivity, expl_path+'continuity/syn/'+expl_type+'/max_sensitivity_one.pkl')

        # TwoLayerSNN
        
        perturbed_explanations = compute_perturbed_explanation(2, X_perturbed, dataset['y_test'], testset_t, expl_path+'continuity/syn/'+expl_type+'/perturbed_explanations_two.pkl', expl_type)
        model_explanations = load_obj(expl_path+'expl_two_syn_nocw_'+expl_type+'.pkl')

        max_sensitivity = max_sensitivity_score(testset_t, model_explanations, perturbed_explanations)
        save_obj(max_sensitivity, expl_path+'continuity/syn/'+expl_type+'/max_sensitivity_two.pkl')

        # ThreeLayerSNN
        #A
        perturbed_explanations = compute_perturbed_explanation(3, X_perturbed, dataset['y_test'], testset_t, expl_path+'continuity/syn/'+expl_type+'/perturbed_explanations_three.pkl', expl_type)
        model_explanations = load_obj(expl_path+'expl_three_syn_nocw_'+expl_type+'.pkl')

        max_sensitivity = max_sensitivity_score(testset_t, model_explanations, perturbed_explanations)
        save_obj(max_sensitivity, expl_path+'continuity/syn/'+expl_type+'/max_sensitivity_three.pkl')

       

