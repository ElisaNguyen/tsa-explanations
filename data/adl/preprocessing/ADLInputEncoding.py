import pandas as pd
import numpy as np


def read_file(path):
    """
    Reads the data from ADL dataset, for both labels and data
    :param path: path to txt file
    returns: dataframe with data or labels
    """
    df = pd.read_table(path, delim_whitespace=True, header=0, skiprows=[1])
    df['Start'] = df['Start'] + ' ' + df['time']
    df['End'] = df['End'] + ' ' + df['time.1']
    df = df.drop(columns=['time', 'time.1'])
    df['Start'] = df['Start'].apply(pd.to_datetime)
    df['End'] = df['End'].apply(pd.to_datetime)
    return df


def encode_sensor(s, data, labels):
    """
    Encodes the sensor activation of a user in spike times across the whole duration of the recorded time in
    time windows of 1 second
    :param labels: Dataframe of labels of a user
    :param data: Dataframe of data of a user
    :param s: String with the location of the sensor/sensor
    :returns: array of spike trains
    """
    spike_times = []

    abs_start = min(data['Start'][0], labels['Start'][0])
    t = abs_start
    abs_end = max(max(data['End']), max(labels['End']))

    if s not in data['Location'].unique():
        return np.zeros(1)

    starts = data[data['Location'] == s]['Start']
    ends = data[data['Location'] == s]['End']
    for start, end in zip(starts, ends):
        while t < start:
            t = t + np.timedelta64(1, 's')
        while start <= t <= end:
            spike = (t - data['Start'][0]).to_numpy() / np.timedelta64(1, 's')
            spike_times.append(int(spike))
            t = t + np.timedelta64(1, 's')

    spike_train = np.zeros(1 + int((abs_end - abs_start).to_numpy() / np.timedelta64(1, 's')))
    for i in spike_times:
        spike_train[i] = 1
    return spike_train


def pad_spike_train(length, spike_train):
    """
    Completing a spike train array of on input dimension to max length with 0s to satisfy input shape
    :param length: the length as int to which the spike train should be filled
    :param spike_train: spike train array to be filled
    :returns: np array spike train with length "length"
    """
    delta = length - len(spike_train)
    filler = np.zeros(delta)
    return np.append(spike_train, filler, axis=0)


def downsample(st, sampling_factor):
    """
    Downsamples a spike train st by a sampling factor
    :param st: Spike train array
    :param sampling_factor: Sampling factor (int)
    :returns: downsampled spike train array
    """
    downsampled_st = np.array([st[i] if i % sampling_factor == 0 else 0 for i in range(len(st))])
    return downsampled_st


def build_input(st, sensors):
    """
    Build input tensor for spiking neural network with dimensions (spike trainxsensors)
    :param sensors: List of sensors
    :param st: spike trains dictionary with all spike trains per sensor
    :returns: input tensor for SNN
    """
    length = np.max([len(spikes) for spikes in st.values()])
    for s in sensors:
        st[s] = pad_spike_train(length, st[s])
    inp = np.transpose(np.array(list(st.values())))
    return inp


def write_labels(data, labels, ts):
    """
    Label each timestep of a user
    :param data: Dataframe of user data
    :param labels: Dataframe of user labels
    :param ts: Range of timesteps starting at 0 in timesteps of 1 second
    :return: Array of labels per timestep to the corresponding data
    """
    t = min(data['Start'][0], labels['Start'][0])
    return_l = []

    for (start, end, a) in zip(labels['Start'], labels['End'], labels['Activity']):
        while t < start:
            return_l.append('NA')
            t = t + np.timedelta64(1, 's')
        while start <= t <= end:
            return_l.append(a)
            t = t + np.timedelta64(1, 's')

    if len(return_l) < len(ts):
        filler = ['NA'] * (len(ts) - len(return_l))
        return_l.extend(filler)
    return return_l