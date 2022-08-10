import random
import sys
import numpy as np
import torch
import os


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


testset_t = load_obj('../../data/synthetic/expl_syn_testset.pkl')
y_true = dataset['y_test'][:, testset_t]

expl_types = ['ns', 's', 'sam']
for expl_type in expl_types:
    with torch.no_grad():
        # get epsilons
        min_attr = -50
        max_attr = 50
        for filename in os.listdir('explanations/{}'.format(expl_type)):
            f = 'explanations/{}/{}'.format(expl_type, filename)
            if os.path.isfile(f):
                max_attr = max(max_attr, get_max_attr(f))
                min_attr = min(min_attr, get_min_attr(f))

        epsilons = get_epsilons(max(max_attr, np.abs(min_attr)))

        for nb_layer in range(3):
            model_explanations = load_obj('explanations/{}/{}L_explanations.pkl'.format(expl_type, nb_layer))
            for i, epsilon in enumerate(epsilons):
                oc_score = output_completeness_score(nb_layer, dataset['X_test'], dataset['y_test'],
                                                     model_explanations, epsilon, testset_t, y_true)
                save_obj(oc_score,
                         'output_completeness/{}/{}L_oc_epsilon{}.pkl'.format(expl_type, nb_layer, i))

            print('Evaluation of output-completeness for {} explanations of SNN-{}L done!'.format(expl_type, nb_layer))