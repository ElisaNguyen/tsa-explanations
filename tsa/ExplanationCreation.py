import random
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

sys.path.insert(1, '../models')
from CoreSNN import sparse_data_generator_from_spikes

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


def tscs(spike_times, tc, model, t, variant):
    """
    Function to calculate the Temporal Spike Contribution Score
    :param variant: TSA variant (string, either 's' or 'ns')
    :param model: SNN model
    :param tc: current time t (int)
    :param t: time to calculate tscs for (int)
    :param spike_times: spike times up until t (Tensor)
    :returns: Tensor of temporal spike contribution scores of all past spikes from spike_times
    """
    diff_t = torch.abs(tc - spike_times)
    score = model.beta ** diff_t
    if len(score) == 0:
        if variant == 's':
            score = torch.tensor([0])
        elif variant == 'ns':
            score = torch.Tensor([-(model.beta ** (tc - t))])
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


def get_weight_contributions(ws):
    """
    Compute weight contributions
    :param ws: all weights of the model
    :returns: weight contribution matrices of same size as ws
    """
    abs_ws = [torch.abs(w) for w in ws]
    w_contributions = []
    for w_a, w in zip(abs_ws, ws):
        min_w = torch.min(w_a)
        max_w = torch.max(w_a)
        w_norm = (w_a - min_w) / (max_w - min_w)
        w_contributions.append(w.sign() * w_norm)
    return w_contributions


def get_spike_trains(spk, nb_units):
    """
    Function to get spike train per input 
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


def attribution_map_mm(model, inp, layer_recs, probs, tc, variant, for_visualization=False):
    """
    Function to compute attribution explanation
    :param variant: TSA variant (string, 's' or 'ns')
    :param for_visualization: Boolean flag whether this is for visualization or full explanation extraction
    :param layer_recs: layer recordings of a model that ran an input (list of dicts)
    :param model: SNN model
    :param inp: spiking input (sparse tensor)
    :param probs: probabilities of the prediction vector (size of c classes) at tc
    :param tc: time of calculating the attribution from the simulation time (current time)
    """
    map = []
    layers_spk_trains = []
    inp_spk_trains = get_spike_trains(inp, model.layer_sizes[0])
    layers_spk_trains.append(inp_spk_trains)
    for rec in layer_recs[:-1]:  # all hidden layers
        h_spk_train = get_spike_trains(rec['spk_rec'].to_sparse(), rec['spk_rec'].shape[-1])
        layers_spk_trains.append(h_spk_train)
    w_contributions = get_weight_contributions(model.weights)

    if for_visualization:
        start_i = tc - 60 if tc > 60 else 0
    else:
        start_i = 0

    for t in range(start_i, tc):
        layers_w_tscs_t = []
        for l in range(model.nb_layers):
            spk_train = layers_spk_trains[l]
            l_tscs_t = [tscs(spk_train[i][spk_train[i] == t], tc, model, t, variant) for i in range(model.layer_sizes[l])]
            l_tscs_t = torch.stack(l_tscs_t).to(device)  # shape (n_layer)
            l_tscs_t = torch.diag(torch.squeeze(l_tscs_t)).type(dtype)  # shape (n_layer, n_layer)
            l_w_tscs_t = torch.matmul(l_tscs_t, w_contributions[l].to(device))  # shape (w[l])
            layers_w_tscs_t.append(l_w_tscs_t)
        w_tscs_t = layers_w_tscs_t[0]
        for l_w_tscs_t in layers_w_tscs_t[1:]:
            w_tscs_t = torch.matmul(w_tscs_t, l_w_tscs_t)
        scores = torch.matmul(w_tscs_t, torch.diag(probs))  # shape (n_input, n_output)
        map.append(scores)
    map = torch.stack(map)  # shape (tc, n_input, n_output)
    map = map.transpose(0, 2)  # shape (n_output, n_input, tc)
    return map


def visualize_data(spike_data, ax, t):
    """
    Function to visualize the spiking data 
    :param spike_data: 2D numpy array with the spike data in the shape (nb_input, tc)
    :param ax: axis to plot it on
    :param tc: current time
    """
    start = t - 60 if t > 60 else 0
    ax.imshow(np.zeros(spike_data.shape), cmap=plt.cm.gray_r)
    ax.grid(color=(229/256,229/256,229/256), linestyle='-', linewidth=0.5)
    ax.set_yticks(range(spike_data.shape[0]))
    ax.set_yticklabels(['Bias', 'Maindoor', 'Cupboard', 'Fridge', 'Shower', 'Toaster', 'Microwave', 'Cooktop',
                        'Seat', 'Toilet', 'Basin', 'Bed', 'Cabinet', 'Door'])
    ax.set_ylabel('Sensors')
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
    visualize_data(data, ax, tc)
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
    # start = len(probs)-60 if len(probs)>60 else 0
    for c in range(nb_outputs):
        #     ax.plot(range(start, len(probs)), probs.t().cpu().numpy()[c][start:], c=color_pallete_hex[c])
        ax.bar(c, height=probs[-1][c].cpu(), color=color_pallete_hex[c])
    ax.set_title('Confidence distribution at timestep ' + str(t), pad=15)
    ax.set_xticklabels([])
    ax.set_xticks(range(nb_outputs))
    ax.set_xlabel('Classes C')
    ax.set_ylabel('P(C|x)')
    ax.legend(titles, bbox_to_anchor=(1.02, 0.5))


def create_cmap(rgb):
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(rgb[0] / 256, 1, N)
    vals[:, 1] = np.linspace(rgb[1] / 256, 1, N)
    vals[:, 2] = np.linspace(rgb[2] / 256, 1, N)
    newcmp = ListedColormap(vals)
    return newcmp


def visualize_attribution_all(inp, attributions, ax, fig, titles, nb_outputs, t):
    """
    Function to visualize the attribution map of one prediction
    :param t: current time - start time (latest time from the data that was run through the network), this can be but must not be the same as tc
    :param nb_outputs: number of outputs (int)
    :param titles: list of class names
    :param fig: matplotlib figure
    :param ax: matplotlib axis to plot on
    :param attributions: explanation attribution map
    :param inp: spiking input (sparse tensor)
    """
    start = t - 60 if t > 60 else 0

    data = np.transpose(inp.to_dense()[0].cpu().numpy())[:, start:t + 1]
    visualize_data(data, ax, t)
    m, am = torch.max(attributions, dim=0)  # get the max class confidence values
    min_attr = torch.min(attributions).detach().cpu().numpy()
    max_attr = torch.max(attributions).detach().cpu().numpy()

    ax.set_title('Feature attribution', pad=15)
    color_pallete_rgb = [(88, 181, 225), (175, 33, 104), (79, 210, 86), (247, 94, 240), (48, 106, 60), (252, 153, 213),
                         (152, 213, 160), (11, 41, 208), (230, 215, 82), (38, 85, 130), (251, 144, 70)]

    for c in range(nb_outputs):
        c_filter = (am == c)
        c_attr = attributions[c].detach().cpu() * c_filter.detach().cpu()
        c_attr = np.ma.masked_where(c_attr == 0, c_attr)
        c_attr = ((c_attr - min_attr) / (max_attr - min_attr)) * 5
        c_attr[c_attr.mask] = -10

        cmap = create_cmap(color_pallete_rgb[c]).reversed()
        cmap.set_under('w', alpha=0)
        # img_overlay = ax.imshow(c_attr[:, start:tc+1], cmap=cmap, alpha=0.7, interpolation='hanning', vmin=-0.05, vmax=5)
        img_overlay = ax.imshow(c_attr, cmap=cmap, alpha=0.7, interpolation='hanning', vmin=-0.05, vmax=5)
        # cax = fig.add_axes([ax.get_position().x1+0.01+(0.06*c),
        # ax.get_position().y0,
        # 0.01,
        # ax.get_position().height])
        # plt.colorbar(img_overlay, label=titles[c], orientation='vertical', cax=cax)


def construct_explanation_all(model, X, y, pred, log_p, layer_recs, t, variant):
    """
    Function to extract the explanation from an input and model and display it
    :param variant: TSA variant (string, 's' or 'ns')
    :param model: SNN model
    :param X: Input (dict form with times and units)
    :param y: ground truth
    :param pred: predictions from the model run
    :param log_p: log probabilities from the model run
    :param layer_recs: layer recordings from the model run
    :param t: timestamp of sample
    :return: explanation attribution map
    """
    prediction = np.squeeze(pred)[-1]
    probs = torch.squeeze(torch.exp(log_p)).t()

    data_generator = sparse_data_generator_from_spikes(X, y, model.bs, model.nb_steps, model.layer_sizes[0],
                                                       model.max_time, shuffle=False)
    X_spikes, _ = next(data_generator)
    attributions = attribution_map_mm(model, X_spikes, layer_recs, probs[-1], t, variant, for_visualization=True)

    fig = plt.figure(tight_layout=False, frameon=False, figsize=(12, 6), dpi=100)
    gs = gridspec.GridSpec(3, 3)
    plt.subplots_adjust(wspace=0.75, hspace=0.5)

    titles = ['Sleeping', 'Toileting', 'Showering', 'Breakfast', 'Grooming',
              'Spare_Time/TV', 'Leaving', 'Lunch', 'Snack', 'Dinner', 'Other']
    color_pallete_hex = ["#58b5e1", "#af2168", "#4fd256", "#f75ef0", "#306a3c", "#fc99d5", "#98d5a0", "#0b29d0",
                         "#e6d752", "#265582", "#fb9046"]

    ax_conf = fig.add_subplot(gs[0, 1:])
    visualize_confidence(ax_conf, probs[:t + 1], color_pallete_hex, model.layer_sizes[-1], titles, t)

    ax_preds = fig.add_subplot(gs[0, 0])
    ax_preds.text(-0.1, 0.25, 'Predicted class: ' + titles[prediction],
                  fontsize=14, ha='left',
                  bbox=dict(facecolor='none', edgecolor=color_pallete_hex[prediction], pad=10.0, linewidth=2))
    ax_preds.text(-0.1, 0.75, 'True label: ' + titles[int(y[0][-1])],
                  fontsize=14, ha='left',
                  bbox=dict(facecolor='none', edgecolor=color_pallete_hex[int(y[0][-1])], pad=10.0, linewidth=2))
    ax_preds.axis('off')

    ax_map = fig.add_subplot(gs[1:, :])
    visualize_attribution_all(X_spikes, attributions, ax_map, fig, titles, model.layer_sizes[-1], t)
    return attributions
