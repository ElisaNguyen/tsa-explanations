# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import clear_output
from sklearn.metrics import balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import random

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


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def live_plot(loss):
    """
    Original from https://github.com/fzenke/spytorch
    live plot during training
    """
    if len(loss) == 1:
        return
    clear_output(wait=True)
    ax = plt.figure(figsize=(3, 2), dpi=150).gca()
    ax.plot(range(1, len(loss) + 1), loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.xaxis.get_major_locator().set_params(integer=True)
    sns.despine()
    plt.show()


def sparse_data_generator_from_spikes(X, y, batch_size, nb_inputs, max_time, shuffle=False):
    """
    This generator takes a spike dataset and generates spiking network input as sparse tensors.
    Original from Friedemann Zenke (https://github.com/fzenke/spytorch), in this version there are changes
    License: http://creativecommons.org/licenses/by/4.0/

    :param X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
    :param y: The labels
    :param batch_size: the batch size (int)
    :param nb_inputs: the dimensionality of the input (int)
    :param max_time: the max time of the dataset (int)
    :param shuffle: whether to shuffle the data or not. Default is False due to non i.i.d. property of time series
    """
    labels_ = y
    number_of_batches = int(np.ceil(len(labels_) / batch_size))
    sample_index = np.arange(len(labels_))

    # compute discrete firing times
    firing_times = X['times']
    units_fired = X['units']

    time_bins = np.linspace(1, max_time,
                            num=max_time)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[sample_index % number_of_batches == counter]
        coo = [[] for i in range(3)]  # coordinate format for sparse tensors in pytorch
        for bc, idx in enumerate(batch_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            batch = [bc for _ in range(len(times))]

            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)  # indeces
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)  # values

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, max_time, nb_inputs])).to(
            device)  # sparse tensor to optimize memory usage
        y_batch = torch.tensor(labels_[batch_index], device=device)
        y_batch = torch.cat((y_batch, (torch.ones(batch_size - len(y_batch), max_time, dtype=torch.long) * 10).to(
            device)))  # if data does not fit in a batch

        yield X_batch.to(device=device), y_batch.to(device=device)

        counter += 1


def conf_interval(perf, n):
    """
    Computes the 95% confidence interval of a performance metric in [0,1]
    :param perf: performance metric
    :param n: sample size
    :return: 95% CI
    """
    z = 1.96  # for 95% CI
    interval = z * np.sqrt((perf * (1 - perf)) / n)
    return interval


class SNN:
    def __init__(self, hyperparams, nb_inputs, nb_outputs, nb_layers, max_time):
        """
        Initialize SNN model with its parameters.

        :param hyperparams: hyperparameters from tuning (dict)
        :param nb_inputs: size of input layer (int)
        :param nb_outputs: size of output layer (int)
        :param nb_layers: amount of computational layers, so all except the input (int)
        :param max_time: max_time of the data samples (int)
        """
        self.nb_layers = nb_layers
        self.max_time = max_time
        self.time_step = hyperparams['time_step']
        self.tau_syn = hyperparams['tau_syn']
        self.tau_mem = hyperparams['tau_mem']
        self.lr = hyperparams['learning_rate']
        self.bs = hyperparams['batch_size']
        self.syns = []
        self.mems = []

        if nb_layers > 1:
            self.layer_sizes = [nb_inputs]
            self.layer_sizes.extend([nb_hidden for nb_hidden in hyperparams['nb_hiddens']])
            self.layer_sizes.append(nb_outputs)
            self.layer_sizes = np.array(self.layer_sizes)
        else:
            self.layer_sizes = np.array([nb_inputs, nb_outputs])

        self.alpha = self._set_alpha()
        self.beta = self._set_beta()
        self.weights = self._set_initial_weights()
        self.spike_fn = SurrGradSpike.apply

    def _set_alpha(self):
        """
        Setter for alpha, the decay parameter for the synaptic currents.
        Original from Friedemann Zenke (https://github.com/fzenke/spytorch), made into a function in this version
        License: http://creativecommons.org/licenses/by/4.0/
        """
        alpha = float(np.exp(-self.time_step / self.tau_syn))
        return alpha

    def _set_beta(self):
        """
        Setter for beta, the decay parameter for the membrane potentials.
        Original from Friedemann Zenke (https://github.com/fzenke/spytorch), made into a function in this version
        License: http://creativecommons.org/licenses/by/4.0/
        """
        beta = float(np.exp(-self.time_step / self.tau_mem))
        return beta

    def _set_initial_weights(self):
        """
        Setter for the network initial weights.
        Original from Friedemann Zenke (https://github.com/fzenke/spytorch), made into a function in this version
        License: http://creativecommons.org/licenses/by/4.0/
        """
        weight_scale = 0.2
        weights = []
        for l in range(self.nb_layers):
            w = torch.empty((self.layer_sizes[l], self.layer_sizes[l + 1]), device=device, dtype=dtype,
                            requires_grad=True)
            torch.nn.init.normal_(w, mean=0.0, std=weight_scale / np.sqrt(self.layer_sizes[l]))
            weights.append(w)
        return weights

    def run_parallel(self, inputs):
        """
        Runs the SNN for 1 batch within one epoch
        Original from Friedemann Zenke (https://github.com/fzenke/spytorch), in this version there are changes
        License: http://creativecommons.org/licenses/by/4.0/
        :param inputs: spiking input to the network
        :returns: membrane potentials and output spikes of all layers
        """
        layer_recs = []
        for l in range(self.nb_layers):
            # lists to record membrane potential and output spikes in the simulation time
            mem_rec = []
            spk_rec = []

            # Compute layer activity
            out = torch.zeros((inputs.shape[0], self.layer_sizes[l + 1]), device=device, dtype=dtype)  # initialization
            # multiplication of input spikes with the weight matrix, this will be fed to the synaptic variable syn
            # and the membrane potential mem
            h = torch.einsum("abc,cd->abd", (inputs, self.weights[l].clone()))  # shape (n_samples, nb_steps, n_outputs)
            for t in range(self.max_time):  # was nb steps before
                mthr = self.mems[l] - 1.0  # subtract the threshold to see if the neurons spike
                out = self.spike_fn(mthr)  # get the layer spiking activity
                rst = out.detach()  # We do not want to backprop through the reset

                self.syns[l] = self.alpha * self.syns[l] + h[:,
                                                           t]  # calculate new input current for the next timestep of
                # the synapsis (PSP), shape (n_samples, n_outputs)
                self.mems[l] = (self.beta * self.mems[l] + self.syns[l]) * (
                        1.0 - rst)  # calculate new membrane potential for the timestep

                mem_rec.append(self.mems[l])  # record the membrane potential
                spk_rec.append(out)  # record the spikes
            # merge the recorded membrane potentials into single tensor
            mem_rec = torch.stack(mem_rec, dim=1)
            # merge output spikes into single tensor
            spk_rec = torch.stack(spk_rec, dim=1)
            layer_recs.append({'mem_rec': mem_rec, 'spk_rec': spk_rec})
            inputs = spk_rec
        return layer_recs

    def train(self, x_data, y_data, path, early_stopping=True):
        """
        Training method, either with early stopping and patience of 10 or for 20 epochs
        Original from Friedemann Zenke (https://github.com/fzenke/spytorch), in this version there are changes
        License: http://creativecommons.org/licenses/by/4.0/
        """
        params = self.weights
        optimizer = optim.Adam(params, lr=self.lr)

        log_softmax_fn = nn.LogSoftmax(dim=1)
        loss_fn = nn.NLLLoss()

        e = 0
        stop_condition = False
        if early_stopping:
            patience = 10
            epochs_no_improve = 0
            min_loss = 200  # initial loss to be overwritten
        else:
            nb_epochs = 20

        loss_hist = []
        # initialize synaptic current and membrane potential
        for l in range(self.nb_layers):
            self.syns.append(torch.zeros((self.bs, self.layer_sizes[l + 1]), device=device, dtype=dtype))
            self.mems.append(torch.zeros((self.bs, self.layer_sizes[l + 1]), device=device, dtype=dtype))
        while not stop_condition:
            batch_counter = 0
            local_loss = []
            for x_local, y_local in sparse_data_generator_from_spikes(x_data, y_data, self.bs,
                                                                      self.layer_sizes[0], self.max_time,
                                                                      shuffle=False):
                layer_recs = self.run_parallel(x_local.to_dense())

                out_mem_rec = layer_recs[-1]['mem_rec']  # classification done on output membrane potential
                output = out_mem_rec.transpose(1, 2)
                log_p_y = log_softmax_fn(output)

                loss = loss_fn(log_p_y, y_local)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                optimizer.step()
                local_loss.append(loss.item())

                # effort to backprop only through batch, detaching from this batch
                for l in range(self.nb_layers):
                    self.syns[l] = self.syns[l].detach().requires_grad_()
                    self.mems[l] = self.mems[l].detach().requires_grad_()
            mean_loss = np.mean(local_loss)
            loss_hist.append(mean_loss)
            roll_amount = int(np.ceil(len(y_data) / self.bs))
            x_data = {'times': np.roll(x_data['times'], -roll_amount, 0),
                      'units': np.roll(x_data['units'], -roll_amount,
                                       0)}  # for the next epoch, assumption that output membrane potential can be new initial membrane potential
            y_data = np.roll(y_data, -roll_amount,
                             0)  # for the next epoch, assumption that output membrane potential can be new initial membrane potential

            if early_stopping:
                if mean_loss < min_loss:
                    min_loss = mean_loss
                    epochs_no_improve = 0
                    torch.save(self.weights, path + "weights_epoch" + str(e) + ".pt")
                else:
                    epochs_no_improve += 1
                    stop_condition = (epochs_no_improve == patience)
            else:
                stop_condition = (e == nb_epochs)
                torch.save(self.weights, path + "weights_epoch" + str(e) + ".pt")

            print("Epoch %i: loss=%.5f" % (e + 1, mean_loss))
            e += 1
        return loss_hist

    def predict(self, x_data, y_data):
        """
        Predicts the class of the input data
        Original from Friedemann Zenke (https://github.com/fzenke/spytorch), in this version there are slight changes
        License: http://creativecommons.org/licenses/by/4.0/
        :param x_data: X
        :param y_data: ground truth
        :returns: y_pred
        """
        batch_size = len(y_data)
        X_spikes, _ = next(sparse_data_generator_from_spikes(x_data, y_data, batch_size,
                                                                  self.layer_sizes[0], self.max_time, shuffle=False))
        layer_recs = self.run_parallel(X_spikes.to_dense())
        out_mem_rec = layer_recs[-1]['mem_rec']  # shape (N (batch_size*nb_steps), 1, n_output)
        output = out_mem_rec.transpose(1, 2)
        log_p_y = nn.LogSoftmax(dim=1)(output)
        _, preds = torch.max(log_p_y, 1)
        return preds.cpu().numpy(), log_p_y.detach(), layer_recs

    def evaluate_loss(self, x_data, y_data):
        """
        Evaluates with NLL loss
        Original from Friedemann Zenke (https://github.com/fzenke/spytorch), made into a function in this version
        License: http://creativecommons.org/licenses/by/4.0/
        :param x_data: X
        :param y_data: ground truth
        :returns: loss
        """
        loss_fn = nn.NLLLoss()
        # reset synaptic currents and membrane potentials
        self.syns = []
        self.mems = []
        for l in range(self.nb_layers):
            self.syns.append(torch.zeros((len(y_data), self.layer_sizes[l + 1]), device=device, dtype=dtype))
            self.mems.append(torch.zeros((len(y_data), self.layer_sizes[l + 1]), device=device, dtype=dtype))
        # Prediction
        _, log_p_y, _ = self.predict(x_data, y_data)
        y_data = torch.tensor(y_data, device=device)
        loss = loss_fn(log_p_y, y_data)
        return loss.item()

    def inference(self, path):
        """
        Sets inference model with parameters specified in the path
        """
        self.weights = torch.load(path, map_location=torch.device(device))

    def evaluate(self, x_data, y_data):
        """
        Evaluates with balanced accuracy at .95 CI
        """
        # reset synaptic currents and membrane potentials
        self.syns = []
        self.mems = []
        for l in range(self.nb_layers):
            self.syns.append(torch.zeros((len(y_data), self.layer_sizes[l + 1]), device=device, dtype=dtype))
            self.mems.append(torch.zeros((len(y_data), self.layer_sizes[l + 1]), device=device, dtype=dtype))
        # Prediction
        y_pred, _, _ = self.predict(x_data, y_data)
        balanced_acc = balanced_accuracy_score(np.ndarray.flatten(np.array(y_data)), np.ndarray.flatten(y_pred))
        ci = conf_interval(balanced_acc, len(np.ndarray.flatten(np.array(y_data))))
        return balanced_acc, ci, y_pred


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).

    Original from Friedemann Zenke (https://github.com/fzenke/spytorch)
    License: http://creativecommons.org/licenses/by/4.0/
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad
