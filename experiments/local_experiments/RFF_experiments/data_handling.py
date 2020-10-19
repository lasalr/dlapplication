import os
from datetime import datetime
from random import shuffle
import numpy as np
import pandas as pd

RANDOM_SEED = 123


def split_dataset(file_path: str, validation_ratio=0.1, test_ratio=0.2, override=False):
    """
    Takes in file path and splits dataset line by line into train, test and validation set based on given ratio
    validation_ratio: fraction of data for validation set
    test_ratio: fraction of data for test set
    The balance is taken as the training set
    """

    # Setting file paths
    validation_file_path = os.path.join(os.path.dirname(file_path), 'split', 'VAL_' + os.path.basename(file_path))
    train_file_path = os.path.join(os.path.dirname(file_path), 'split', 'TRAIN_' + os.path.basename(file_path))
    test_file_path = os.path.join(os.path.dirname(file_path), 'split', 'TEST_' + os.path.basename(file_path))

    if not override and os.path.isfile(validation_file_path) and os.path.isfile(train_file_path)\
            and os.path.isfile(test_file_path):
        print('All files are already present. No splitting done.')
        return

    # Checking length of dataset and allocating indices at random for split
    with open(file_path, 'r') as f:
        data = f.readlines()

    length = len(data)
    shuffle(data)
    val_end_idx = int(length * validation_ratio)
    test_end_idx = int(val_end_idx + (length * test_ratio))

    # Split data Validation, Train, Test split
    val_data = data[: val_end_idx]
    test_data = data[val_end_idx: test_end_idx]
    train_data = data[test_end_idx:]

    # Clear original file
    data = None

    # Create folder to store datasets
    if not os.path.exists(os.path.join(os.path.dirname(file_path), 'split')):
        os.makedirs(os.path.join(os.path.dirname(file_path), 'split'))

    # Removing old files
    if override:
        with open(validation_file_path, 'w') as validation_file_writer:
            validation_file_writer.writelines(val_data)
        with open(test_file_path, 'w') as test_file_writer:
            test_file_writer.writelines(test_data)
        with open(train_file_path, 'w') as train_file_writer:
            train_file_writer.writelines(train_data)

    # Saving only files not present
    else:
        if not os.path.isfile(validation_file_path):
            with open(validation_file_path, 'w') as validation_file_writer:
                validation_file_writer.writelines(val_data)
        if not os.path.isfile(test_file_path):
            with open(test_file_path, 'w') as test_file_writer:
                test_file_writer.writelines(test_data)
        if not os.path.isfile(train_file_path):
            with open(train_file_path, 'w') as train_file_writer:
                train_file_writer.writelines(train_data)


def load_data(path: str, label_col: int, d: int):
    df = pd.read_csv(filepath_or_buffer=path, names=[x for x in range(0, d + 1)])
    labels = df.iloc[:, label_col].to_numpy(dtype=np.int32)
    features = df.drop(df.columns[label_col], axis=1).to_numpy(dtype=np.float64)
    return features, labels

def write_experiment(path, name: str, start_time, experiment_list: [dict]):
    out_string = 'Experiments conducted: {}\n'.format(len(experiment_list))
    out_string += 'Start time: {}\n'.format(str(start_time))
    out_string += 'End time: {}\n'.format(str(datetime.now()))
    out_string += '\n--------------Best result--------------\n'
    for k, v in sorted(experiment_list, key=lambda ky: ky['ROC_AUC'], reverse=True)[0].items():
        out_string += str(k) + '= ' + str(v) + '\n'

    out_string += '\n--------------All results--------------\n'
    for experiment in experiment_list:
        for k, v in experiment.items():
            out_string += str(k) + '= ' + str(v) + '\n'

    out_file = os.path.join(path, 'Results_' + str(name) + str(start_time).replace(':', '_').replace(' ', '_') + '.txt')
    with open(out_file, 'w') as f:
        f.write(out_string)


def write_csv(path: str, name: str, start_time, results: dict, sortby_col: str, sort_ascending=True):
    out_file = os.path.join(path, 'Results_' + str(name) + str(start_time).replace(':', '_').replace(' ', '_') + '.csv')
    pd.DataFrame(results).sort_values(sortby_col).to_csv(out_file, index=False)
