import functools
import itertools
import operator
import os
import statistics
import sys
from datetime import datetime

import numpy as np
import scipy.special
from sklearn.preprocessing import StandardScaler

from experiments.local_experiments.RFF_experiments.data_handling import split_dataset

sys.path.append("../../../..")
sys.path.append("../../../../../dlplatform")


class DataGenerator:
    """
    Synthetic data generator
    """
    RANDOM_STATE = 123

    def __init__(self, poly_deg, size, dim, data_folder):
        self.dim = dim
        self.poly_deg = poly_deg
        self.size = size
        self.min_coef = -10
        self.max_coef = 10
        self.data_folder = data_folder

    def __call__(self):
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)
        self.data_file_path = os.path.join(self.data_folder, 'SYNTHETIC_DATA_' +
                                           str(datetime.now()).replace(':', '_').replace(' ', '_')[:19] + '.csv')
        self.val_file_path = os.path.join(self.data_folder, os.path.dirname(self.data_file_path), 'split',
                                          'VAL_' + os.path.basename(self.data_file_path))
        self.train_data_path = os.path.join(self.data_folder, os.path.dirname(self.data_file_path), 'split',
                                            'TRAIN_' + os.path.basename(self.data_file_path))
        self.test_data_path = os.path.join(self.data_folder, os.path.dirname(self.data_file_path), 'split',
                                           'TEST_' + os.path.basename(self.data_file_path))

        np.random.seed = DataGenerator.RANDOM_STATE
        coeffs = self.generate_coeffs()
        bias = np.random.uniform(low=-1, high=1)
        size = self.size

        X, Yval = [], []
        for i in range(size):
            x, y_val = self.generate_datapoint(coeffs, bias)
            X.append(x)
            Yval.append(y_val)

        # Scaling y
        scaler = StandardScaler()
        y_val = scaler.fit_transform(y_val.reshape(-1, 1))

        # Adding Gaussian noise to result
        epsy = np.random.normal(loc=0.0, scale=0.3, size=y_val.shape)
        y_val = y_val + epsy

        theta = statistics.median(Yval)
        Y = [1 if y_val >= theta else -1 for y_val in Yval]
        print(np.average(Y))

        X = np.array(X)
        # Scaling X
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        # adding Gaussian noise to features
        epsX = np.random.normal(loc=0.0, scale=0.3, size=X.shape)
        X = X + epsX

        Y = np.array(Y)
        Yval = np.array(Yval)
        print(X.shape, Y.shape, Yval.shape)

        data = np.hstack((Y.reshape((Y.shape[0], 1)), X))
        print(data.shape)

        np.savetxt(fname=self.data_file_path, X=data, comments='', delimiter=',')
        print('Splitting dataset...')
        split_dataset(file_path=os.path.abspath(self.data_file_path))  # Does not save if file is present
        return os.path.abspath(self.data_file_path)

    def generate_datapoint(self, coeffs, bias):
        # Sample X (vector of DIM dimensions)
        X = np.random.uniform(low=-10, high=10, size=self.dim)

        # Create polynomial combinations e.g. (x1^3, x2^3, x1^2 * x2, ...)
        #     (x1^2, x2^2, x3^2, x1*x2, ...) ... (...)
        poly = [itertools.combinations_with_replacement(X, r) for r in
                range(1, self.poly_deg + 1)]

        # Multiplying each tuple from above
        multiplied_polys_temp = [[functools.reduce(operator.mul, list(tup), 1) for tup in p] for
                                 p in poly]

        # Multiplying final values with coefficients
        multiplied_poly = [functools.reduce(operator.mul, val_list + [c], 1) for c, val_list in
                           zip(coeffs, multiplied_polys_temp)]

        # Calculating result
        y_val = bias + sum(multiplied_poly)

        # Return row of data
        return X, y_val

    def generate_coeffs(self):
        # Calculating number of coefficients
        n_coeff = sum([scipy.special.comb(self.dim, r, repetition=True) for
                       r in range(1, self.poly_deg + 1)])

        # Creating randomly sampled coefficients
        return np.random.uniform(low=self.min_coef, high=self.max_coef, size=int(n_coeff))
