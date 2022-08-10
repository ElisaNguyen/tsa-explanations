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
with torch.no_grad():
    # generate the perturbed data
    X_perturbed = generate_perturbed_data(dataset['X_test'], testset_t,
                                            'continuity/perturbed_X.pkl')

    for nb_layer in range(3):
        for expl_type in expl_types:
            print('Evaluating continuity for {} explanations of {}L-SNN...'.format(expl_type, nb_layer))
            perturbed_explanations = compute_perturbed_explanation(nb_layer, X_perturbed, dataset['y_test'],
                                                                   testset_t, expl_type,
                                                                   'continuity/{}/perturbed_explanations_{}.pkl'.format(
                                                                       expl_type, nb_layer))
            model_explanations = load_obj('explanations/{}/{}L_explanations.pkl'.format(expl_type, nb_layer))
            max_sensitivity = max_sensitivity_score(testset_t, model_explanations, perturbed_explanations)
            save_obj(max_sensitivity, 'continuity/{}/max_sensitivity_{}.pkl'.format(expl_type, nb_layer))
        print('Continuity evaluation of SNN-{}L is done!'.format(nb_layer))

       

