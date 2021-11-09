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
        start_t = t-3600 if t>3600 else 0 
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


def compute_perturbed_explanation(nb_layers, X_perturbed, y_data, testset_t, path):
    """
    Computes max sensitivity score as the maximum change in the explanation after perturbation. 
    """
    perturbed_explanations = {}
    for t in tqdm(testset_t):
        start_t = t-3600 if t>3600 else 0 
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
        data_generator = sparse_data_generator_from_spikes(X_perturbed[t], y, 1, 14, max_time)
        X_p_spikes, _ = next(data_generator)
        
        attribution_p = attribution_map_mm(model, X_p_spikes, layer_recs_p, probs_p[-1], max_time)
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


def generate_sensitivity_baseline(testset_t, X_perturbed, y_data, savepath):
    """
    generates random explanation with values in between min and max attribution of the explanations of a model
    """
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

    baseline_explanations = {}
    for t in tqdm(testset_t): 
        # get the relevant part of the dataset, this is done for performance reasons
        start_t = t-3600 if t>=3600 else 0
        X = X_perturbed[t]
        y = y_data[:, start_t:t]
        max_time = t-start_t
        
        data_generator = sparse_data_generator_from_spikes(X, y, len(y), max_time, 14, max_time, shuffle=False)
        X_spikes, _ = next(data_generator)
        
        baseline_explanations[t] = (assign_random_attribution(X_spikes.to_dense()[0].t(), min_attr, max_attr).detach(), y[0][-1])
        save_obj(baseline_explanations, savepath)
    return baseline_explanations
    

def assign_random_attribution(X_spikes, min_attr, max_attr):
    e = torch.zeros(X_spikes.shape).to(device)
    e[X_spikes!=0] = torch.Tensor(np.random.uniform(min_attr, max_attr, X_spikes[X_spikes!=0].shape)).to(device)
    return e
    

A_testset_t = load_obj('/local/work/enguyen/data/quantitative_test_t_A.pkl')
B_testset_t = load_obj('/local/work/enguyen/data/quantitative_test_t_B.pkl')
A_y_true = dataset['y_test_A'][:, A_testset_t]
B_y_true = dataset['y_test_B'][:, B_testset_t]

with torch.no_grad():
    # generate the perturbed data
    X_perturbed_A = generate_perturbed_data(dataset['X_test_A'], A_testset_t, '/local/work/enguyen/evaluation/sensitivity/naturally_perturbed_X_A.pkl')
    X_perturbed_B = generate_perturbed_data(dataset['X_test_B'], B_testset_t, '/local/work/enguyen/evaluation/sensitivity/naturally_perturbed_X_B.pkl')
    
    # generate baseline
    baseline_p_explanations_A = generate_sensitivity_baseline(A_testset_t, X_perturbed_A, dataset['y_test_A'], '/local/work/enguyen/evaluation/sensitivity/baseline_p_explanations_A.pkl')
    baseline_p_explanations_B = generate_sensitivity_baseline(B_testset_t, X_perturbed_B, dataset['y_test_B'], '/local/work/enguyen/evaluation/sensitivity/baseline_p_explanations_B.pkl')

    # TSA-S
    # OneLayerSNN
    # A
    perturbed_explanations = compute_perturbed_explanation(1, X_perturbed_A, dataset['y_test_A'], A_testset_t, '/local/work/enguyen/evaluation/sensitivity/tsa-s/perturbed_explanations_oneA.pkl')
    model_explanations = load_obj('/local/work/enguyen/evaluation/tsa-s/onelayer_explanations_A.pkl')
    max_sensitivity = max_sensitivity_score(A_testset_t, model_explanations, perturbed_explanations)
    save_obj(max_sensitivity, '/local/work/enguyen/evaluation/sensitivity/tsa-s/max_sensitivity_oneA.pkl')

    # B
    perturbed_explanations = compute_perturbed_explanation(1, X_perturbed_B, dataset['y_test_B'], B_testset_t, '/local/work/enguyen/evaluation/sensitivity/tsa-s/perturbed_explanations_oneB.pkl')
    model_explanations = load_obj('/local/work/enguyen/evaluation/tsa-s/onelayer_explanations_B.pkl')
    max_sensitivity = max_sensitivity_score(B_testset_t, model_explanations, perturbed_explanations)
    save_obj(max_sensitivity, '/local/work/enguyen/evaluation/sensitivity/tsa-s/max_sensitivity_oneB.pkl')

    # TwoLayerSNN
    # A
    perturbed_explanations = compute_perturbed_explanation(2, X_perturbed_A, dataset['y_test_A'], A_testset_t, '/local/work/enguyen/evaluation/sensitivity/tsa-s/perturbed_explanations_twoA.pkl')
    model_explanations = load_obj('/local/work/enguyen/evaluation/tsa-s/twolayer_explanations_A.pkl')
    max_sensitivity = max_sensitivity_score(A_testset_t, model_explanations, perturbed_explanations)
    save_obj(max_sensitivity, '/local/work/enguyen/evaluation/sensitivity/tsa-s/max_sensitivity_twoA.pkl')
    
    # B
    perturbed_explanations = compute_perturbed_explanation(2, X_perturbed_B, dataset['y_test_B'], B_testset_t, '/local/work/enguyen/evaluation/sensitivity/tsa-s/perturbed_explanations_twoB.pkl')
    model_explanations = load_obj('/local/work/enguyen/evaluation/tsa-s/twolayer_explanations_B.pkl')
    max_sensitivity = max_sensitivity_score(B_testset_t, model_explanations, perturbed_explanations)
    save_obj(max_sensitivity, '/local/work/enguyen/evaluation/sensitivity/tsa-s/max_sensitivity_twoB.pkl')
    
    # ThreeLayerSNN
    # A
    perturbed_explanations = compute_perturbed_explanation(3, X_perturbed_A, dataset['y_test_A'], A_testset_t, '/local/work/enguyen/evaluation/sensitivity/tsa-s/perturbed_explanations_threeA.pkl')
    model_explanations = load_obj('/local/work/enguyen/evaluation/tsa-s/threelayer_explanations_A.pkl')
    max_sensitivity = max_sensitivity_score(A_testset_t, model_explanations, perturbed_explanations)
    save_obj(max_sensitivity, '/local/work/enguyen/evaluation/sensitivity/tsa-s/max_sensitivity_threeA.pkl')

    #B
    perturbed_explanations = compute_perturbed_explanation(3, X_perturbed_B, dataset['y_test_B'], B_testset_t, '/local/work/enguyen/evaluation/sensitivity/tsa-s/perturbed_explanations_threeB.pkl')
    model_explanations = load_obj('/local/work/enguyen/evaluation/tsa-s/threelayer_explanations_B.pkl')
    max_sensitivity = max_sensitivity_score(B_testset_t, model_explanations, perturbed_explanations)
    save_obj(max_sensitivity, '/local/work/enguyen/evaluation/sensitivity/tsa-s/max_sensitivity_threeB.pkl')

    # TSA-NS
    # OneLayerSNN
    # A
    perturbed_explanations = compute_perturbed_explanation(1, X_perturbed_A, dataset['y_test_A'], A_testset_t,
                                                           '/local/work/enguyen/evaluation/sensitivity/tsa-ns/perturbed_explanations_oneA.pkl')
    model_explanations = load_obj('/local/work/enguyen/evaluation/tsa-ns/onelayer_explanations_A.pkl')
    max_sensitivity = max_sensitivity_score(A_testset_t, model_explanations, perturbed_explanations)
    save_obj(max_sensitivity, '/local/work/enguyen/evaluation/sensitivity/tsa-ns/max_sensitivity_oneA.pkl')

    # B
    perturbed_explanations = compute_perturbed_explanation(1, X_perturbed_B, dataset['y_test_B'], B_testset_t,
                                                           '/local/work/enguyen/evaluation/sensitivity/tsa-ns/perturbed_explanations_oneB.pkl')
    model_explanations = load_obj('/local/work/enguyen/evaluation/tsa-ns/onelayer_explanations_B.pkl')
    max_sensitivity = max_sensitivity_score(B_testset_t, model_explanations, perturbed_explanations)
    save_obj(max_sensitivity, '/local/work/enguyen/evaluation/sensitivity/tsa-ns/max_sensitivity_oneB.pkl')

    # TwoLayerSNN
    # A
    perturbed_explanations = compute_perturbed_explanation(2, X_perturbed_A, dataset['y_test_A'], A_testset_t,
                                                           '/local/work/enguyen/evaluation/sensitivity/tsa-ns/perturbed_explanations_twoA.pkl')
    model_explanations = load_obj('/local/work/enguyen/evaluation/tsa-ns/twolayer_explanations_A.pkl')
    max_sensitivity = max_sensitivity_score(A_testset_t, model_explanations, perturbed_explanations)
    save_obj(max_sensitivity, '/local/work/enguyen/evaluation/sensitivity/tsa-ns/max_sensitivity_twoA.pkl')

    # B
    perturbed_explanations = compute_perturbed_explanation(2, X_perturbed_B, dataset['y_test_B'], B_testset_t,
                                                           '/local/work/enguyen/evaluation/sensitivity/tsa-ns/perturbed_explanations_twoB.pkl')
    model_explanations = load_obj('/local/work/enguyen/evaluation/tsa-ns/twolayer_explanations_B.pkl')
    max_sensitivity = max_sensitivity_score(B_testset_t, model_explanations, perturbed_explanations)
    save_obj(max_sensitivity, '/local/work/enguyen/evaluation/sensitivity/tsa-ns/max_sensitivity_twoB.pkl')

    # ThreeLayerSNN
    # A
    perturbed_explanations = compute_perturbed_explanation(3, X_perturbed_A, dataset['y_test_A'], A_testset_t,
                                                           '/local/work/enguyen/evaluation/sensitivity/tsa-ns/perturbed_explanations_threeA.pkl')
    model_explanations = load_obj('/local/work/enguyen/evaluation/tsa-ns/threelayer_explanations_A.pkl')
    max_sensitivity = max_sensitivity_score(A_testset_t, model_explanations, perturbed_explanations)
    save_obj(max_sensitivity, '/local/work/enguyen/evaluation/sensitivity/tsa-ns/max_sensitivity_threeA.pkl')

    # B
    perturbed_explanations = compute_perturbed_explanation(3, X_perturbed_B, dataset['y_test_B'], B_testset_t,
                                                           '/local/work/enguyen/evaluation/sensitivity/tsa-ns/perturbed_explanations_threeB.pkl')
    model_explanations = load_obj('/local/work/enguyen/evaluation/tsa-ns/threelayer_explanations_B.pkl')
    max_sensitivity = max_sensitivity_score(B_testset_t, model_explanations, perturbed_explanations)
    save_obj(max_sensitivity, '/local/work/enguyen/evaluation/sensitivity/tsa-ns/max_sensitivity_threeB.pkl')

    #### Baselines
    max_sensitivity = max_sensititivity_score(A_testset_t, baseline_explanations_A, baseline_p_explanations_A)
    save_obj(max_sensitivity, '/local/work/enguyen/zero/evaluation/sensitivity/max_sensitivity_baseline_A.pkl')
    max_sensitivity = max_sensititivity_score(B_testset_t, baseline_explanations_B, baseline_p_explanations_B)
    save_obj(max_sensitivity, '/local/work/enguyen/zero/evaluation/sensitivity/max_sensitivity_baseline_B.pkl')
    