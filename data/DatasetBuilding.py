import numpy as np
import pandas as pd
import torch
import random

torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

dtype = torch.float

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def train_test_split(t, df):
    """
    Splits time series dataset into train, validation and test set with a 60-20-20 split.
    Targeted to be used in an apply function to a Series object.
    :param t: current timestep (int)
    :param df: pandas Dataframe containing the data
    :returns: subset categorization as string
    """
    if t < (len(df) * 0.6):
        return 'train'
    elif t < (len(df) * 0.8):
        return 'val'
    else:
        return 'test'


def extract_even_samples(duration, df):
    """
    Returns spike trains of all input neurons from the input set (df) cut into samples of a certain duration.
    Dataset is on 1 second granularity
    :param duration: duration of the samples in seconds (int)
    :param df: data as pandas dataframe 
    :returns: data and labels in arrays (same length)
    """
    data = []
    labels = []
    start_i = 0
    padding = duration - (len(df) % duration) if duration > 1 and duration != len(df) else 0
    df_pad = pd.DataFrame(index=range(padding), columns=df.columns)
    df_pad['Class'] = 10
    df_pad.loc[:, df_pad.columns[:14]] = 0
    df = df.append(df_pad, ignore_index=True)
    while start_i < len(df):
        end_i = start_i + duration - 1
        data.append(df.loc[start_i:end_i, df.columns[:14]].to_numpy())
        labels.append(list(df['Class'][start_i:end_i + 1]))
        start_i = end_i + 1
    return data, np.array(labels)


def get_times_and_units(samples):
    """
  Transforms sensor data from channels into a dictionary of the spike times and the
  corresponding units that spiked.
  :param samples: sensor data samples (2D numpy arrays)
  :returns: dictionary of times and units
  """
    data = {'times': [], 'units': []}
    for times, units in pd.Series(samples).apply(lambda x: np.where(x == 1)):
        data['times'].append(times)
        data['units'].append(units)
    data['times'] = np.array(data['times'])
    data['units'] = np.array(data['units'])
    return data


def generate_dataset(df, duration):
    """
    Generates specified subset of the data cut into time series of a certain duration
    :param duration: maximum duration of the samples
    :param df: dataframe with data (with subset)
    :returns: dataset with times and units, labels as a dictionary
    """
    samples, labels = extract_even_samples(duration, df)
    data = get_times_and_units(samples)
    return data, labels
