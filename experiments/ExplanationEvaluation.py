# -*- coding: utf-8 -*-
import random
import sys
import numpy as np
import torch
import tqdm
from sklearn import metrics

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)

dtype = torch.float

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

sys.path.insert(1, '../models')
from CoreSNN import *

"""# Helper functions"""


def sample_n_testset(labels, n):
    """
    samples n timestamps of each class from data
    :param labels: y_data
    :param n: int
    :returns: sampled timestamps
    """
    test_set_t = []
    classes = np.unique(np.squeeze(labels))
    for c in classes:
        c_timesteps = np.where(np.squeeze(labels) == c)[0]
        c_sample = np.random.choice(c_timesteps, n)
        test_set_t.append(c_sample)
    test_set_t = np.sort(np.ndarray.flatten(np.array(test_set_t)))
    return test_set_t


def initiate_adl_model(nb_layers, t):
    """
    Function that initiates a SNN model with nb_layers which runs data of duration t, only defined for 3 layers
    :param nb_layers: (int) to define the number of layers
    :param t: max_time and nb_steps is defined by this (int)
    """
    nb_inputs = 14
    nb_outputs = 11
    if nb_layers == 1:
        params_1L = load_obj('../models/best_params_1L.pkl')
        model = SNN(hyperparams=params_1L,
                    nb_inputs=nb_inputs,
                    nb_outputs=nb_outputs,
                    nb_layers=1,
                    nb_steps=t,
                    max_time=t)

        model.inference('../models/weights_1L_epoch4.pt')
        return model
    elif nb_layers == 2:
        params_2L = load_obj('../model/best_params_2L.pkl')
        params_2L['nb_hiddens'] = [params_2L['nb_hidden']]
        model = SNN(hyperparams=params_2L,
                    nb_inputs=nb_inputs,
                    nb_outputs=nb_outputs,
                    nb_layers=2,
                    nb_steps=t,
                    max_time=t)

        model.inference('../models/weights_2L_epoch63.pt')
        return model
    elif nb_layers == 3:
        params_3L = load_obj('../models/best_params_3L.pkl')
        params_3L['nb_hiddens'] = [params_3L['nb_hidden1'], params_3L['nb_hidden2']]
        model = SNN(hyperparams=params_3L,
                    nb_inputs=nb_inputs,
                    nb_outputs=nb_outputs,
                    nb_layers=3,
                    nb_steps=t,
                    max_time=t)

        model.inference('../models/weights_3L_epoch48.pt')
        return model


def initiate_syn_model(nb_layers, t):
    """
    Function that initiates a SNN model with nb_layers which runs data of duration t, only defined for 3 layers for the synthetic dataset
    :param nb_layers: (int) to define the number of layers
    :param t: max_time and nb_steps is defined by this (int)
    """
    nb_inputs = 3
    nb_outputs = 4
    hyperparams = {'time_step': 0.001,
               'tau_syn': 0.01,
               'tau_mem': 0.001,
               'optimizer': optim.Adam,
               'learning_rate': 0.01,
               'batch_size': 128,
               'nb_hiddens': 10}
    if nb_layers == 1:
        params_onelayer = hyperparams
        model = SNN(hyperparams=params_onelayer,
                          nb_inputs=nb_inputs,
                          nb_outputs=nb_outputs,
                          nb_layers=1,
                          nb_steps=t,
                          max_time=t)

        model.inference('../models/synthetic/weights_1L.pt')
        return model
    elif nb_layers == 2:
        params_twolayer = hyperparams
        params_twolayer['nb_hiddens'] = [hyperparams['nb_hiddens']]
        model = SNN(hyperparams=params_twolayer,
                          nb_inputs=nb_inputs,
                          nb_outputs=nb_outputs,
                          nb_layers=2,
                          nb_steps=t,
                          max_time=t)

        model.inference('../models/synthetic/weights_2L.pt')
        return model
    elif nb_layers == 3:
        params_threelayer = hyperparams
        params_twolayer['nb_hiddens'] = [hyperparams['nb_hiddens'], hyperparams['nb_hiddens']]
        model = SNN(hyperparams=params_threelayer,
                          nb_inputs=nb_inputs,
                          nb_outputs=nb_outputs,
                          nb_layers=3,
                          nb_steps=t,
                          max_time=t)
        model.inference('../models/synthetic/weights_3L.pt')
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
    segments_d_pos, segments_t_pos = torch.where(e > 0)
    segments_d_neg, segments_t_neg = torch.where(e < 0)
    for segments_d, segments_t in zip([segments_d_pos, segments_d_neg], [segments_t_pos, segments_t_neg]):
        attributing_sensors = segments_d.unique()

        # identification is done per sensor dimension
        for sensor in attributing_sensors:
            i_sensor = torch.where(segments_d == sensor)
            sensor_segment = segments_t[i_sensor]
            t_diff = torch.diff(sensor_segment)
            # idea find the subsegments and then identify the segments around the max there
            subsegments = []
            subsegment_start = 0
            for segment_edge in torch.where(t_diff != 1)[0]:
                subsegments.append(sensor_segment[subsegment_start:segment_edge + 1])
                subsegment_start = segment_edge + 1
            subsegments.append(sensor_segment[subsegment_start:])

            # each subsegment is now contiguous, so that the max attribution can be identified first
            for subsegment in subsegments:
                segment_start = subsegment[0]
                segment_end = subsegment[-1] + 1

                t_max_attr = torch.argmax(e[sensor, subsegment])
                t_end_max = t_max_attr + (max_window_size / 2) if t_max_attr > max_window_size else torch.tensor(
                    [max_window_size])
                t_end_max = torch.clip(t_end_max, segment_start, segment_end).to(device)
                t_start_max = torch.clip(t_end_max - max_window_size, segment_start, segment_end).to(device)
                feature_segments.append(
                    (sensor.cpu().numpy(), int(t_start_max.cpu().numpy()), int(t_end_max.cpu().numpy())))

                left_segments_in_subsegment_existing = t_start_max != segment_start
                right_segments_in_subsegment_existing = t_end_max != segment_end

                t_start = t_start_max
                while left_segments_in_subsegment_existing:
                    # go left of the max attribution
                    t_end = t_start
                    t_start = torch.clip(t_end - max_window_size, segment_start, segment_end)
                    feature_segments.append(
                        (sensor.cpu().numpy(), int(t_start.cpu().numpy()), int(t_end.cpu().numpy())))
                    left_segments_in_subsegment_existing = t_start != segment_start

                t_end = t_end_max
                while right_segments_in_subsegment_existing:
                    # go right when all of the lefts are identified
                    t_start = t_end
                    t_end = torch.clip(t_start + max_window_size, segment_start, segment_end)
                    feature_segments.append(
                        (sensor.cpu().numpy(), int(t_start.cpu().numpy()), int(t_end.cpu().numpy())))
                    right_segments_in_subsegment_existing = t_end != segment_end
    return feature_segments


def rank_segments(e, feature_segments):
    """
    rank segments of e
    :param e: explanation e
    :param feature_segments: unranked feature segments (list of (sensor, t_start, t_end))
    :returns: sorted list of the feature segments with the mean score as well
    """
    scores = [torch.mean(e[d, t_start:t_end]).detach().cpu().numpy() for d, t_start, t_end in feature_segments]
    ranked_feature_segments = sorted(zip(scores, feature_segments), reverse=True)  # sort by scores in descending order
    return ranked_feature_segments


def predict_from_spikes(model, X_spikes):
    """
    Helper function to predict from spiking format instead of X format
    :param model: SNN model
    :param X_spikes: spiking input
    :returns: y_pred, log probabilities, layer recs (mem_rec and spk_rec)
    """
    # reset membrane potential and synaptic currents
    model.syns = []
    model.mems = []
    for l in range(model.nb_layers):
        model.syns.append(torch.zeros((1, model.layer_sizes[l + 1]), device=device, dtype=dtype))
        model.mems.append(torch.zeros((1, model.layer_sizes[l + 1]), device=device, dtype=dtype))
    layer_recs = model.run_parallel(X_spikes.to_dense())
    out_mem_rec = layer_recs[-1]['mem_rec']
    output = out_mem_rec.transpose(1, 2)
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
    e[e != 0] = torch.Tensor(np.random.uniform(min_attr, max_attr, e[e != 0].shape)).to(device)
    return e


# Functions for the evaluation of correctness
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


# Functions for the evaluation of output-completeness
def identify_high_attribution(ranked_fs, epsilon):
    """
    filters the feature segments to those that are contributing more than epsilon
    :param ranked_fs: list of feature segments with their mean score
    :param epsilon: threshold for high attribution
    :return: feature segments that are >epsilon and the sensor dimensions that belong to them
    """
    high_attribution_segments = []
    attributing_sensors = []
    for fs in ranked_fs:
        if np.abs(fs[0]) > epsilon:
            high_attribution_segments.append(fs)
            if fs[1][0] not in attributing_sensors:
                attributing_sensors.append(fs[1][0])
    return high_attribution_segments, attributing_sensors


def get_time_segments(sensor, segments):
    """
    Function to extract the start timestamps and end timestamps sorted from early to late from a feature segment
    :param sensor: sensor dimension of segment (int)
    :param segments: feature segments to extract the timestamps for
    :returns: starts and ends in sorted lists
    """
    starts = []
    ends = []
    for _, segment in segments:
        if segment[0] == sensor:
            starts.append(segment[1])
            ends.append(segment[2])
    starts_sorted = [start for start, _ in sorted(zip(starts, ends))]
    ends_sorted = [end for _, end in sorted(zip(starts, ends))]
    return starts_sorted, ends_sorted


def perturb_background(attributing_sensors, X_spikes, segments, nb_inputs):
    """
    Function to perturb only the background and not the high attributing segments randomly
    :param attributing_sensors: attributing sensors
    :param X_spikes: spiking input
    :param segments: highly attributing segments
    :returns: perturbed X_spikes
    """
    X_spikes_perturbed = X_spikes.to_dense()[0, :, :].t()
    for sensor in range(nb_inputs):
        if sensor in attributing_sensors:
            starts, ends = get_time_segments(sensor, segments)

            # identify the background data (non high attributing)
            background_data = [X_spikes_perturbed[sensor][:starts[0]]]
            for next_start, end in zip(starts[1:], ends[:-1]):
                background_data.append(X_spikes_perturbed[sensor][end:next_start])
            background_data.append(X_spikes_perturbed[sensor][ends[-1]:])
            background_data = torch.cat(background_data)

            # randomly permute the background
            shuffle_idx = torch.randperm(len(background_data))
            background_data = background_data[shuffle_idx]
            perturbed_data = [background_data[:starts[0]]]
            background_data = background_data[starts[0]:]
            for i in range(len(starts) - 1):
                perturbed_data.append(X_spikes_perturbed[sensor][starts[i]:ends[i]])
                perturbed_data.append(background_data[:(starts[i + 1] - ends[i])])
                background_data = background_data[(starts[i + 1] - ends[i]):]
            perturbed_data.append(X_spikes_perturbed[sensor][starts[-1]:ends[-1]])
            perturbed_data.append(background_data)
            perturbed_data = torch.cat(perturbed_data)
            X_spikes_perturbed[sensor] = perturbed_data
        else:
            shuffle_idx = torch.randperm(X_spikes.shape[1])
            X_spikes_perturbed[sensor] = X_spikes_perturbed[sensor][shuffle_idx]
    X_spikes_perturbed = X_spikes_perturbed.t()
    X_spikes_perturbed = torch.unsqueeze(X_spikes_perturbed, 0)
    return X_spikes_perturbed.to_sparse()


def output_completeness_score(nb_layers, nb_inputs, X_data, y_data, model_explanations, epsilon, testset_t, y_true):
    """
    Computes the attribution sufficiency score as the balanced accuracy of the model's predictions on data with perturbed background and the ground truth
    :param nb_inputs: input dimensionality
    :param y_true: ground truth labels to compare against
    :param nb_layers: number of layers of the SNN
    :param model_explanations: extracted explanations and X_spikes dictionary
    :param epsilon: threshold for high attribution
    :param testset_t: timestamps to test
    :returns: attribution sufficiency score and y_pred_bgperturbed
    """
    y_pred_bgperturbed = []
    y_preds = []
    model = initiate_model(nb_layers, 1)
    for t in tqdm(testset_t):
        start_t = t - 3600 if t >= 3600 else 0
        model.nb_steps = t - start_t
        model.max_time = t - start_t

        e, prediction = model_explanations[t]
        y_preds.append(prediction)

        # get the relevant part of the dataset, this is done for performance reasons
        X = {'times': X_data['times'][:, np.where((X_data['times'] >= start_t) & (X_data['times'] < t))[1]] - start_t,
             'units': X_data['units'][:, np.where((X_data['times'] >= start_t) & (X_data['times'] < t))[1]]}
        y = y_data[:, start_t:t]
        data_generator = sparse_data_generator_from_spikes(X, y, len(y), model.nb_steps, model.layer_sizes[0],
                                                           model.max_time, shuffle=False)
        X_spikes, _ = next(data_generator)

        fs = segment_features(e)
        fs_scores = rank_segments(e, fs)
        high_attribution_segments, attributing_sensors = identify_high_attribution(fs_scores, epsilon)
        X_spikes_perturbed = perturb_background(attributing_sensors, X_spikes, high_attribution_segments, nb_inputs)

        y_pred_perturbed_here, _, _ = predict_from_spikes(model, X_spikes_perturbed)
        y_pred_bgperturbed.append(y_pred_perturbed_here[0][-1])
    score = metrics.balanced_accuracy_score(y_preds, y_pred_bgperturbed)
    return score, y_preds, y_pred_bgperturbed


def get_epsilons(max_attr):
    """
    Given all explanations of one SNN model across both data subjects, find the epsilons at 25,
    50 and 75% of the attribution range of positive attributions
    :param max_attr: maximum absolute attribution value
    :return:
    """
    epsilons = [(0), (0.25 * max_attr), (0.5 * max_attr), (0.75 * max_attr)]
    return epsilons


# Functions for the evaluation of continuity
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
    Function to randomly draw the perturbation variables that determine the natural duration perturbation
    :param sensor_spikes: Original activity spike_train of one sensor
    :return: Determined perturbation in terms of what kind, where and by how much
    """
    perturbation = np.random.choice(['shorten', 'lengthen'])
    perturbation_loc = np.random.choice(['start', 'end'])
    perturbation_size = np.random.choice(np.arange(np.round(0.1 * len(sensor_spikes)) + 1))
    return perturbation, perturbation_loc, perturbation_size


def perform_perturbation_continuity(max_time, sensor_spikes, sensor, X_times, X_units):
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
        X_perturbed_times, X_perturbed_units = perform_perturbation_continuity(max_time, sensor_spikes, sensor, X_perturbed_times,
                                                                               X_perturbed_units)
    else:
        edges = np.where(np.diff(sensor_spikes) != 1)[0] + 1
        start_edge = 0
        for edge in edges:
            X_perturbed_times, X_perturbed_units = perform_perturbation_continuity(max_time, sensor_spikes[start_edge:edge],
                                                                                   sensor, X_perturbed_times, X_perturbed_units)
            start_edge = edge
        X_perturbed_times, X_perturbed_units = perform_perturbation_continuity(max_time, sensor_spikes[edges[-1]:], sensor,
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
            attribution_p = sam(model, X_p_spikes, layer_recs_p, probs_p[-1], max_time, tsa_variant='s')
        else:
            attribution_p = tsa(model, X_p_spikes, layer_recs_p, probs_p[-1], max_time, explanation_type)
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
