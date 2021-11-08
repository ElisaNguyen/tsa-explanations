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

sys.path.insert(1, '/local/work/enguyen')
from CoreSNN import *

"""# Helper functions"""


def sample_n_testset(labels,n):
    """
    samples n timestamps of each class from data
    :param labels: y_data
    :param n: int
    :returns: sampled timestamps
    """
    test_set_t = []
    classes = np.unique(np.squeeze(labels))
    for c in classes:
        c_timesteps = np.where(np.squeeze(labels)==c)[0]
        c_sample = np.random.choice(c_timesteps, n)
        test_set_t.append(c_sample)
    test_set_t = np.sort(np.ndarray.flatten(np.array(test_set_t)))
    return test_set_t


def initiate_model(nb_layers, t):
    """
    Function that initiates a SNN model with nb_layers which runs data of duration t, only defined for 3 layers
    :param nb_layers: (int) to define the number of layers
    :param t: max_time and nb_steps is defined by this (int)
    """
    nb_inputs = 14
    nb_outputs = 11
    if nb_layers == 1:
        params_onelayer = load_obj('/local/work/enguyen/tuning/results_onelayersnn/best_params.pkl')
        model = SNN(hyperparams=params_onelayer, 
                          nb_inputs=nb_inputs, 
                          nb_outputs=nb_outputs, 
                          nb_layers=1, 
                          nb_steps=t, 
                          max_time=t)

        model.inference('/local/work/enguyen/training/onelayersnn/weights_epoch4.pt')
        return model
    elif nb_layers == 2:
        params_twolayer = load_obj('/local/work/enguyen/tuning/results_twolayersnn/best_params.pkl')
        params_twolayer['nb_hiddens'] = [params_twolayer['nb_hidden']]
        model = SNN(hyperparams=params_twolayer, 
                          nb_inputs=nb_inputs, 
                          nb_outputs=nb_outputs, 
                          nb_layers=2, 
                          nb_steps=t, 
                          max_time=t)

        model.inference('/local/work/enguyen/training/twolayersnn/weights_epoch63.pt')
        return model
    elif nb_layers == 3:
        params_threelayer = load_obj('/local/work/enguyen/tuning/results_threelayersnn/best_params.pkl')
        params_threelayer['nb_hiddens'] = [params_threelayer['nb_hidden1'], params_threelayer['nb_hidden2']]
        model = SNN(hyperparams=params_threelayer, 
                          nb_inputs=nb_inputs, 
                          nb_outputs=nb_outputs, 
                          nb_layers=3, 
                          nb_steps=t, 
                          max_time=t)

        model.inference('/local/work/enguyen/training/threelayersnn/weights_epoch48.pt')
        return model


def segment_features(e, max_window_size=10): 
    """
    Segment features based on a max window, starting around the highest attribution values in the explanation e.
    :param e: explanation of the predicted class
    :param max_window_size: maximum size of the segment
    :returns: list of segmented features with one feature being characterized by (sensor, t_start, t_end)
    """
    # identify the feature segments
    feature_segments = []
    
    # identify the dimension and timesteps of the attributing features, first pos then neg so that segments are not mixed
    segments_d_pos, segments_t_pos = torch.where(e>0)
    segments_d_neg, segments_t_neg = torch.where(e<0)
    for segments_d, segments_t in zip([segments_d_pos, segments_d_neg], [segments_t_pos, segments_t_neg]):
        attributing_sensors = segments_d.unique()

        # identification is done per sensor dimension
        for sensor in attributing_sensors: 
            i_sensor = torch.where(segments_d == sensor)
            sensor_segment = segments_t[i_sensor]
            t_diff = torch.diff(sensor_segment)
            #idea find the subsegments and then identify the segments around the max there
            subsegments = []
            subsegment_start = 0
            for segment_edge in torch.where(t_diff!=1)[0]:
                subsegments.append(sensor_segment[subsegment_start:segment_edge+1])
                subsegment_start = segment_edge+1
            subsegments.append(sensor_segment[subsegment_start:])

            #each subsegment is now contiguous, so that the max attribution can be identified first
            for subsegment in subsegments:
                segment_start = subsegment[0]
                segment_end = subsegment[-1]+1

                t_max_attr = torch.argmax(e[sensor, subsegment])
                t_end_max = t_max_attr+(max_window_size/2) if t_max_attr>max_window_size else torch.tensor([max_window_size])
                t_end_max = torch.clip(t_end_max, segment_start, segment_end).to(device)
                t_start_max = torch.clip(t_end_max - max_window_size, segment_start, segment_end).to(device)
                feature_segments.append((sensor.cpu().numpy(), int(t_start_max.cpu().numpy()), int(t_end_max.cpu().numpy())))
                
                left_segments_in_subsegment_existing = t_start_max!=segment_start
                right_segments_in_subsegment_existing = t_end_max!=segment_end

                t_start = t_start_max
                while left_segments_in_subsegment_existing:
                    #go left of the max attribution
                    t_end = t_start
                    t_start = torch.clip(t_end-max_window_size, segment_start, segment_end)
                    feature_segments.append((sensor.cpu().numpy(), int(t_start.cpu().numpy()), int(t_end.cpu().numpy())))
                    left_segments_in_subsegment_existing = t_start!=segment_start

                t_end = t_end_max
                while right_segments_in_subsegment_existing:
                    #go right when all of the lefts are identified
                    t_start = t_end
                    t_end = torch.clip(t_start + max_window_size, segment_start, segment_end)
                    feature_segments.append((sensor.cpu().numpy(), int(t_start.cpu().numpy()), int(t_end.cpu().numpy())))
                    right_segments_in_subsegment_existing = t_end!=segment_end
    return feature_segments


def rank_segments(e, feature_segments):
    """
    rank segments of e
    :param e: explanation e
    :param feature_segments: unranked feature segments (list of (sensor, t_start, t_end))
    :returns: sorted list of the feature segments with the mean score as well
    """
    scores = [torch.mean(e[d, t_start:t_end]).detach().cpu().numpy() for d, t_start, t_end in feature_segments]
    ranked_feature_segments = sorted(zip(scores, feature_segments), reverse=True) # sort by scores in descending order
    return ranked_feature_segments


def predict_from_spikes(model, X_spikes): 
    """
    Helper function to predict from spiking format instead of X format
    :param model: SNN model
    :param X_spikes: spiking input
    :returns: y_pred, log probabilities, layer recs (mem_rec and spk_rec)
    """
    #reset membrane potential and synaptic currents
    model.syns=[]
    model.mems=[]
    for l in range(model.nb_layers):
        model.syns.append(torch.zeros((1,model.layer_sizes[l+1]), device=device, dtype=dtype))
        model.mems.append(torch.zeros((1,model.layer_sizes[l+1]), device=device, dtype=dtype))
    layer_recs = model.run_parallel(X_spikes.to_dense())
    out_mem_rec = layer_recs[-1]['mem_rec']
    output = out_mem_rec.transpose(1,2)
    log_p_y = nn.LogSoftmax(dim=1)(output)
    _, preds = torch.max(log_p_y, 1)
    return preds.detach().cpu().numpy(), log_p_y.detach(), layer_recs
    

def get_max_attr(model_explanations):
    max_attr = 0
    for key in model_explanations.keys():
        local_max = torch.max(model_explanations[key][0]).cpu().numpy()
        max_attr = local_max if local_max > max_attr else max_attr
    return max_attr
    
    
def get_min_attr(model_explanations):
    min_attr = 0
    for key in model_explanations.keys():
        local_min = torch.min(model_explanations[key][0]).cpu().numpy()
        min_attr = local_min if local_min < min_attr else min_attr
    return min_attr
    

def assign_random_attribution(e, min_attr, max_attr):
    e[e!=0] = torch.Tensor(np.random.uniform(min_attr, max_attr, e[e!=0].shape)).to(device)
    return e