import os
from random import shuffle

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

    # Removing old files if explicitly requested
    elif override:
        if os.path.isfile(validation_file_path):
            os.remove(validation_file_path)
        if os.path.isfile(train_file_path):
            os.remove(train_file_path)
        if os.path.isfile(test_file_path):
            os.remove(test_file_path)

    with open(validation_file_path, 'a') as validation_file_writer:
        validation_file_writer.writelines(val_data)
    with open(test_file_path, 'a') as test_file_writer:
        test_file_writer.writelines(test_data)
    with open(train_file_path, 'a') as train_file_writer:
        train_file_writer.writelines(train_data)


if __name__ == '__main__':
    split_dataset(file_path='../../../data/SUSY/SUSY.csv')