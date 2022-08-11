# -*- coding: utf-8 -*-
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


sys.path.insert(1, '../../models')
from CoreSNN import *
sys.path.insert(1, '../')
from ExplanationCreation import *
from ExplanationEvaluation import *

# Load data
syn_data = load_obj('../../data/syn_data.pkl')

testset_expl = load_obj('../../data/expl_syn_testset.pkl')

y_true = syn_data['y_test'][:, testset_expl]

# Fixed parameters, nbsteps and max time correspond to the set duration (for testing, first we consider the validation set and not the test set)
nb_inputs  = 3
nb_outputs = 4


def extract_explanations_for_quantitative_analysis(testset_t, explanation_type, nb_layers, X_data, y_data, filename):
    """
    Helper function to extract the X_spikes, explanations (attribution maps) and the prediction for a model
    :param testset_t: the timestamps to be run and extract explanations for
    :param explanation_type: string defining the explanation type: s, ns, ns2 or ncs
    :param nb_layers: amount of layers of the model
    :param X_data: data in the dictionary times, units form
    :param y_data: labels
    :param filename: string of the filename to save the information under
    """
    testset_explanations = {}
    for t in tqdm(testset_t): 
        # get the relevant part of the dataset, this is done for performance reasons
        start_t = t-1000 if t>=1000 else 0
        X = {'times': X_data['times'][:, np.where((X_data['times']>=start_t) & (X_data['times']<t))[1]]-start_t, 
             'units': X_data['units'][:, np.where((X_data['times']>=start_t) & (X_data['times']<t))[1]]}
        y = y_data[:, start_t:t]

        model = initiate_syn_model(nb_layers, (t-start_t))
        
        #reset synaptic currents and membrane potentials to fit the data duration 
        model.syns=[]
        model.mems=[]
        for l in range(model.nb_layers):
            model.syns.append(torch.zeros((len(y),model.layer_sizes[l+1]), device=device, dtype=dtype))
            model.mems.append(torch.zeros((len(y),model.layer_sizes[l+1]), device=device, dtype=dtype))

        y_pred, log_p_y, layer_recs = model.predict(X, y)
        probs = torch.exp(log_p_y)[0].t()
        data_generator = sparse_data_generator_from_spikes(X, y, len(y), model.nb_steps, model.layer_sizes[0], model.max_time, shuffle=False)
        X_spikes, _ = next(data_generator)

        if explanation_type == 'sam':
            attribution = sam(model, X_spikes, layer_recs, probs[-1], t-start_t, tsa_variant='s')
        else: 
            attribution = tsa(model, X_spikes, layer_recs, probs[-1], t-start_t, explanation_type)
        prediction = y_pred[0][-1]
        e = attribution[prediction]

        testset_explanations[t] = (e.detach(), prediction)
        save_obj(testset_explanations, 'explanations/' + filename + '.pkl')


expl_types = ['s', 'ns', 'sam']

for nb_layer in range(3):
    for expl_type in expl_types:
        extract_explanations_for_quantitative_analysis(testset_expl, nb_layer, syn_data['X_test'], syn_data['y_test'],
                                                       expl_type, '{}/{}L_explanations'.format(expl_type, nb_layer))
