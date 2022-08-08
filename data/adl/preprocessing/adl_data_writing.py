import pandas as pd
import numpy as np
import os
from ADLInputEncoding import read_file, downsample, encode_sensor, build_input, write_labels

os.chdir('../data/raw/UCI ADL Binary Dataset')
df_labels_A = read_file('OrdonezA_ADLs.txt')
df_labels_B = read_file('OrdonezB_ADLs.txt')
df_data_A = read_file('OrdonezA_Sensors.txt')
df_data_B = read_file('OrdonezB_Sensors.txt') \
    # comment: A and B do not have the same sensor list
sensors = list(set().union(df_data_A['Location'].unique(), df_data_B['Location'].unique()))

st_A = {}
st_B = {}
for sensor in sensors:
    st_A[sensor] = downsample(encode_sensor(sensor, df_data_A, df_labels_A), 0)
    st_B[sensor] = downsample(encode_sensor(sensor, df_data_B, df_labels_B), 0)

timestamps_A = range(int((df_data_A['End'][408] - df_data_A['Start'][0]).to_numpy() / np.timedelta64(1, 's')) + 1)
abs_start_B = min(df_data_B['Start'][0], df_labels_B['Start'][0])
abs_end_B = max(max(df_data_B['End']), max(df_labels_B['End']))
timestamps_B = range(int((abs_end_B - abs_start_B).to_numpy() / np.timedelta64(1, 's')) + 1)

os.chdir('../../../..')
df_labelled_A = pd.DataFrame(build_input(st_A, sensors), columns=sensors)
df_labelled_A['t'] = timestamps_A
df_labelled_A['Label'] = write_labels(df_data_A, df_labels_A, timestamps_A)
df_labelled_A.to_csv('OrdonezA.csv', index=False)

df_labelled_B = pd.DataFrame(build_input(st_B, sensors), columns=sensors)
df_labelled_B['t'] = timestamps_B
df_labelled_B['Label'] = write_labels(df_data_B, df_labels_B, timestamps_B)
df_labelled_B.to_csv('OrdonezB.csv', index=False)
