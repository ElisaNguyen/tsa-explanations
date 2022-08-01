# -*- coding: utf-8 -*-

import random
import numpy as np
import pickle
import pandas as pd
from data import DatasetBuilding as dsb

random.seed(123)
np.random.seed(123)


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


"""### Import data and transform it to right format"""

labels = ['Sleeping', 'Toileting', 'Showering', 'Breakfast', 'Grooming',
          'Spare_Time/TV', 'Leaving', 'Lunch', 'Snack', 'Dinner']


def set_class_from_label(label):
    if label in labels:
        return labels.index(label)
    else:
        return 10  # other class


# specific to the binary ADL dataset
df_A = pd.read_csv('./OrdonezA.csv')
df_B = pd.read_csv('./OrdonezB.csv')

# add bias
df_A.insert(0, 'Bias', 1)
df_B.insert(0, 'Bias', 1)

df_A['Class'] = df_A['Label'].apply(lambda x: set_class_from_label(x))
df_B['Class'] = df_B['Label'].apply(lambda x: set_class_from_label(x))

df_A['t'] = df_A.index
df_B['t'] = df_B.index

df_A['set'] = df_A['t'].apply(lambda x: dsb.train_test_split(x, df_A))
df_B['set'] = df_B['t'].apply(lambda x: dsb.train_test_split(x, df_B))

"""# Generate dataset"""

duration = 900

X_train_A, y_train_A = dsb.generate_dataset(df_A[df_A['set'] == 'train'], duration)
X_val_A, y_val_A = dsb.generate_dataset(df_A[df_A['set'] == 'val'], duration)
X_test_A, y_test_A = dsb.generate_dataset(df_A[df_A['set'] == 'test'], duration)
X_train_B, y_train_B = dsb.generate_dataset(df_B[df_B['set'] == 'train'], duration)
X_val_B, y_val_B = dsb.generate_dataset(df_B[df_B['set'] == 'val'], duration)
X_test_B, y_test_B = dsb.generate_dataset(df_B[df_B['set'] == 'test'], duration)

X_train = {'times': np.append(X_train_A['times'], X_train_B['times']),
           'units': np.append(X_train_A['units'], X_train_B['units'])}
X_val = {'times': np.append(X_val_A['times'], X_val_B['times']),
         'units': np.append(X_val_A['units'], X_val_B['units'])}
X_test = {'times': np.append(X_test_A['times'], X_test_B['times']),
          'units': np.append(X_test_A['units'], X_test_B['units'])}

dataset = {'X_train': X_train,
           'y_train': np.append(y_train_A, y_train_B, axis=0),
           'X_val': X_val,
           'y_val': np.append(y_val_A, y_val_B, axis=0),
           'X_test': X_test,
           'y_test': np.append(y_test_A, y_test_B, axis=0)}

save_obj(dataset, 'dataset900.pkl')

X_train_A, y_train_A = dsb.generate_dataset(df_A[df_A['set'] == 'train'], len(df_A[df_A['set'] == 'train']))
X_val_A, y_val_A = dsb.generate_dataset(df_A[df_A['set'] == 'val'], len(df_A[df_A['set'] == 'val']))
X_test_A, y_test_A = dsb.generate_dataset(df_A[df_A['set'] == 'test'], len(df_A[df_A['set'] == 'test']))
X_train_B, y_train_B = dsb.generate_dataset(df_B[df_B['set'] == 'train'], len(df_B[df_B['set'] == 'train']))
X_val_B, y_val_B = dsb.generate_dataset(df_B[df_B['set'] == 'val'], len(df_B[df_B['set'] == 'val']))
X_test_B, y_test_B = dsb.generate_dataset(df_B[df_B['set'] == 'test'], len(df_B[df_B['set'] == 'test']))

dataset_max = {'X_train_A': X_train_A,
               'y_train_A': y_train_A,
               'X_train_B': X_train_B,
               'y_train_B': y_train_B,
               'X_val_A': X_val_A,
               'y_val_A': y_val_A,
               'X_val_B': X_val_B,
               'y_val_B': y_val_B,
               'X_test_A': X_test_A,
               'y_test_A': y_test_A,
               'X_test_B': X_test_B,
               'y_test_B': y_test_B}

save_obj(dataset_max, 'dataset_max.pkl')
