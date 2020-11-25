import re
import pandas as pd
import itertools
import operator
import functools
import scipy.special
import statistics
import os
import sys
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from DLplatform.parameters.vectorParameters import VectorParameter

sys.path.append("../../../..")
sys.path.append("../../../../../dlplatform")

from experiments.local_experiments.RFF_experiments.data_handling import load_data, split_dataset, write_csv
from DLplatform.aggregating import RadonPoint, Average

RANDOM_STATE = 123
RESULTS_FOLDER = './Results/'
DATA_FILE_PATH = 'SYNTHETIC_DATA_' + str(datetime.now()).replace(':', '_').replace(' ', '_')[:19] + '.csv'
VAL_FILE_PATH = os.path.join(os.path.dirname(DATA_FILE_PATH), 'split', 'VAL_' + os.path.basename(DATA_FILE_PATH))
TRAIN_DATA_PATH = os.path.join(os.path.dirname(DATA_FILE_PATH), 'split', 'TRAIN_' + os.path.basename(DATA_FILE_PATH))
TEST_DATA_PATH = os.path.join(os.path.dirname(DATA_FILE_PATH), 'split', 'TEST_' + os.path.basename(DATA_FILE_PATH))
DATASET_NAME = 'SYNTHETIC5'
DATASET_SIZE = 5_000_000
DIM = 5
POLY_DEG = 3
DATA_LABEL_COL = 0
TUNE_DATA_FRACTION = 0.01




