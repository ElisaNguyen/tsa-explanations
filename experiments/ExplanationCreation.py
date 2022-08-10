import random
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)
plt.style.use('ggplot')

dtype = torch.float
# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)


def spike_contribution(spike_times, tc, model, t, variant, layer_size=None):
    """
    Function to calculate the spike contribution (referred to as N in the paper)
    :param layer_size: size of layer for which to compute the tscs (int)
    :param variant: TSA expl_type (string, either 's' or 'ns')
    :param model: SNN model
    :param tc: current time t (int)
    :param t: time to calculate the spike contribution for (int)
    :param spike_times: spike times up until t (Tensor)
    :returns: Tensor of scores of all past spikes from spike_times
    """
    diff_t = torch.abs(tc - spike_times)
    score = model.beta ** diff_t
    if len(score) == 0:
        if variant == 's':
            score = torch.tensor([0])
        elif variant == 'ns':
            score = torch.Tensor([(-1/layer_size)*(model.beta ** (tc - t))])
    return score


def spk_times_per_unit(spk, units, unit):
    """
    Gets the spikes per units 
    :param spk: spikes formatted like 'times'
    :param units: corresponding 'units'
    :param unit: the unit that the spike train should be retrieved for
    :returns: tensor of the spike times 
    """
    ix = np.where(units.cpu().numpy() == unit)
    spk_times = spk.cpu().numpy()[ix]
    return torch.Tensor(spk_times)


def get_spike_trains(spk, nb_units):
    """
    Function to get spike train per input dimension
    :param spk: spikes
    :param nb_units: amount of inpute dimensions
    :return: spike trains per input in a nb_units x T format
    """
    times = spk._indices()[1]
    units = spk._indices()[2]

    spk_trains = []
    for unit in range(nb_units):
        unit_spk_train = spk_times_per_unit(times, units, unit)
        spk_trains.append(unit_spk_train)
    return spk_trains


def tsa(model, inp, layer_recs, probs, tc, variant, for_visualization=False):
    """
    Function to compute TSA explanation
    :param variant: TSA expl_type (string, 's' or 'ns')
    :param for_visualization: Boolean flag whether this is for visualization or full explanation extraction
    :param layer_recs: layer recordings of a model that ran an input (list of dicts)
    :param model: SNN model
    :param inp: spiking input (sparse tensor)
    :param probs: probabilities of the prediction vector (size of c classes) at tc
    :param tc: time of calculating the attribution from the simulation time (current time)
    """
    tsa_map = []
    layers_spk_trains = []
    inp_spk_trains = get_spike_trains(inp, model.layer_sizes[0])
    layers_spk_trains.append(inp_spk_trains)
    for rec in layer_recs[:-1]:  # all hidden layers
        h_spk_train = get_spike_trains(rec['spk_rec'].to_sparse(), rec['spk_rec'].shape[-1])
        layers_spk_trains.append(h_spk_train)
    w_contributions = model.weights  # weight contribution

    if for_visualization:  # if this is for visualization, compute only the last minute to speed up computation
        start_i = tc - 60 if tc > 60 else 0
    else:
        start_i = 0

    for t in range(start_i, tc):
        layers_w_spk_contr_t = []  # initialize to store N_w per layer
        for l in range(model.nb_layers):
            spk_train = layers_spk_trains[l]  # get the spike trains from the layer
            l_spk_contr_t = [spike_contribution(spk_train[i][spk_train[i] == t], tc, model, t, variant, layer_size=model.layer_sizes[l])
                        for i in range(model.layer_sizes[l])]  # compute the spike train contribution N(t)
            l_spk_contr_t = torch.stack(l_spk_contr_t).to(device)  # shape (n_layer)
            l_spk_contr_t = torch.diag(torch.squeeze(l_spk_contr_t)).type(dtype)  # shape (n_layer, n_layer)
            l_w_spk_contr_t = torch.matmul(l_spk_contr_t, w_contributions[l].to(device))  # shape (w[l]), compute the weighted spike contribution N_w(t) of a layer
            layers_w_spk_contr_t.append(l_w_spk_contr_t)
        w_spk_contr_t = layers_w_spk_contr_t[0]
        for l_w_spk_contr_t in layers_w_spk_contr_t[1:]:
            w_spk_contr_t = torch.matmul(w_spk_contr_t, l_w_spk_contr_t) # forward pass of the spike time and weights contribution
        scores = torch.matmul(w_spk_contr_t, torch.diag(probs))  # shape (n_input, n_output), forward pass to the last layer (multiplication with the classification confidence)
        tsa_map.append(scores)
    tsa_map = torch.stack(tsa_map)  # shape (tc, n_input, n_output)
    tsa_map = tsa_map.transpose(0, 2)  # shape (n_output, n_input, tc)
    return tsa_map


def sam(model, inp, layer_recs, probs, tc, for_visualization=False):
    """
    Function to compute attribution without considering the weight
    (in other words, weight contributions =1, having no impact on the scores when multiplying)
    Application of Kim and Panda (2021)'s work.
    :param for_visualization: Boolean flag whether this is for visualization or full explanation extraction
    :param layer_recs: layer recordings of a model that ran an input (list of dicts)
    :param model:SNN model
    :param inp: input spikes
    :param probs: probabilities of the prediction vector (size of c classes) at tc
    :param tc: time of calculating the attribution from the simulation time (current time)
    """
    sam_map = []
    layers_spk_trains = []
    inp_spk_trains = get_spike_trains(inp, model.layer_sizes[0])
    layers_spk_trains.append(inp_spk_trains)
    for rec in layer_recs[:-1]:  # all hidden layers
        h_spk_train = get_spike_trains(rec['spk_rec'].to_sparse(), rec['spk_rec'].shape[-1])
        layers_spk_trains.append(h_spk_train)

    if for_visualization:
        start_i = tc - 40 if tc > 40 else 0
    else:
        start_i = 0

    for t in range(start_i, tc):
        layers_w_spk_contr_t = []
        for l in range(model.nb_layers):
            spk_train = layers_spk_trains[l]
            l_spk_contr_t = [spike_contribution(spk_train[i][spk_train[i] == t], tc, model, t, 's') for i in
                        range(model.layer_sizes[l])]
            l_spk_contr_t = torch.stack(l_spk_contr_t).to(device)  # shape (n_layer)
            l_spk_contr_t = torch.diag(torch.squeeze(l_spk_contr_t)).type(dtype)  # shape (n_layer, n_layer)
            l_w_spk_contr_t = torch.matmul(l_spk_contr_t, torch.ones(model.weights[l].shape))  # shape (w[l])
            layers_w_spk_contr_t.append(l_w_spk_contr_t)
        w_spk_contr_t = layers_w_spk_contr_t[0]
        for l_w_spk_contr_t in layers_w_spk_contr_t[1:]:
            w_spk_contr_t = torch.matmul(w_spk_contr_t, l_w_spk_contr_t)
        scores = torch.matmul(w_spk_contr_t, torch.diag(probs).to(device))  # shape (n_input, n_output)
        sam_map.append(scores)
    sam_map = torch.stack(sam_map)  # shape (tc, n_input, n_output)
    sam_map = sam_map.transpose(0, 2)  # shape (n_output, n_input, tc)
    return sam_map


def visualize_data(spike_data, ax, t, titles):
    """
    Function to visualize the spiking data 
    :param spike_data: 2D numpy array with the spike data in the shape (nb_input, tc)
    :param ax: axis to plot it on
    :param tc: current time
    :param titles: array of input dimension titles (strings)
    """
    start = t - 60 if t > 60 else 0
    ax.imshow(np.zeros(spike_data.shape), cmap=plt.cm.gray_r)
    ax.grid(color=(229/256,229/256,229/256), linestyle='-', linewidth=0.5)
    ax.set_yticks(range(spike_data.shape[0]))
    ax.set_yticklabels(titles)
    ax.set_ylabel('Input sensors')
    ax.set_xticks(range(0, spike_data.shape[1]))
    ax.set_xticklabels(range(start + 1, t + 1), rotation=315, ha='left')
    ax.set_xlabel('Time (sec)')

    s_y, s_x = np.where(spike_data != 0)

    for xx, yy in zip(s_x, s_y):
        amount = spike_data[yy, xx]
        for spike in range(int(amount)):
            offset = spike / amount
            spike_plot = Rectangle((xx + offset, yy - 0.5), 0.005, 1, color='black')
            ax.add_patch(spike_plot)


def visualize_attribution_class(inp, e, tc, ax, fig, titles):
    """
    Function to visualize the attribution map of one class for one prediction
    :param titles: list of class names
    :param fig: matplotlib figure
    :param ax: matplotlib axis to plot on
    :param e: explanation (2D feature attribution map)
    :param inp: input data (spikes)
    :param tc: current time (int)
    """
    start = tc - 60 if tc > 60 else 0
    attributions, c = e
    data = np.transpose(inp.to_dense()[0].cpu().numpy())[:, start:tc + 1]
    visualize_data(data, ax, tc, titles)
    denom = torch.max(attributions)

    img_overlay = ax.imshow((attributions / denom).detach().cpu()[:, start:tc + 1], cmap=plt.cm.seismic, alpha=.7,
                            interpolation='hanning', vmin=-1, vmax=1)
    cax = fig.add_axes([ax.get_position().x1 + 0.01,
                        ax.get_position().y0,
                        0.01,
                        ax.get_position().height])
    cbar = plt.colorbar(img_overlay, cax=cax)
    cbar.set_ticks([])
    ax.set_title('Feature attribution\n' + titles[c], pad=15)


def visualize_confidence(ax, probs, color_pallete_hex, nb_outputs, titles, t):
    """
    Function to visualize the class confidences up to tc
    :param titles: list of class names
    :param nb_outputs: number of classes (int)
    :param color_pallete_hex: color palette in hex code with as many colors as classes
    :param ax: axis to plot on
    :param probs: classification probabilities to plot up to tc
    """
    for c in range(nb_outputs):
        ax.bar(c, height=probs[-1][c].cpu(), color=color_pallete_hex[c])
    ax.set_title('Confidence distribution at timestep ' + str(t), pad=15)
    ax.set_xticklabels([])
    ax.set_xticks(range(nb_outputs))
    ax.set_xlabel('Classes C')
    ax.set_ylabel('P(C|x)')
    ax.legend(titles, bbox_to_anchor=(1.02, 0.5))