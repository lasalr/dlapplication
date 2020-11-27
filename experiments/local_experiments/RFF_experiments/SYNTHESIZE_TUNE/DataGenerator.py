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

    def __init__(self, poly_deg, size, dim, data_folder, xy_noise_scale=None, x_range=None, bias_range=None):
        self.dim = dim
        self.poly_deg = poly_deg
        self.size = size
        self.min_coef = -10
        self.max_coef = 10
        self.data_folder = data_folder
        if xy_noise_scale is None:
            self.xy_noise_scale = [0.1, 0.1]
        else:
            self.xy_noise_scale = xy_noise_scale
        if x_range is None:
            self.x_range = [-10, 10]
        else:
            self.x_range = x_range
        if bias_range is None:
            self.bias_range = [-1, 1]
        else:
            self.bias_range = bias_range

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
        bias = np.random.uniform(low=self.bias_range[0], high=self.bias_range[1])
        size = self.size

        X, Y_values = [], []
        for i in range(size):
            x_point, y_point = self.generate_datapoint(coeffs, bias)
            X.append(x_point)
            Y_values.append(y_point)

        # Scaling y
        # scaler = StandardScaler()
        Y_values = np.array(Y_values)
        # Y_values = scaler.fit_transform(Y_values.reshape(-1, 1))

        # Adding Gaussian noise to result
        eps_y = np.random.normal(loc=0.0, scale=self.xy_noise_scale[1], size=Y_values.shape)

        theta = statistics.median(Y_values)
        print('SD of Y_values={} with Y_values.shape={}'.format(np.std(a=Y_values), Y_values.shape))
        before_Y_values_shape = Y_values.shape
        # Add multiplicative noise to y
        Y_values = Y_values * (1 + eps_y)
        # Add additive noise to y
        # Y_values = Y_values + eps_y
        after_Y_values_shape = Y_values.shape
        assert before_Y_values_shape == after_Y_values_shape

        Y = [1 if y_val >= theta else -1 for y_val in Y_values]
        print('Average label value, np.average(Y)={}'.format(np.average(Y)))
        assert np.average(Y) < 0.3
        X = np.array(X)
        # Scaling X
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        # adding Gaussian noise to features
        eps_X = np.random.normal(loc=0.0, scale=self.xy_noise_scale[0], size=X.shape)

        before_X_shape = X.shape
        # Add multiplicative noise to X
        X = X * (1 + eps_X)
        after_X_shape = X.shape
        assert before_X_shape == after_X_shape

        Y = np.array(Y)
        print('X.shape={}, Y.shape={}, Y_values.shape={}'.format(X.shape, Y.shape, Y_values.shape))

        data = np.hstack((Y.reshape((Y.shape[0], 1)), X))
        print('data.shape={}'.format(data.shape))

        np.savetxt(fname=self.data_file_path, X=data, comments='', delimiter=',')
        print('Splitting dataset...')
        split_dataset(file_path=os.path.abspath(self.data_file_path))  # Does not save if file is present
        return os.path.abspath(self.data_file_path)

    def generate_datapoint(self, coeffs, bias):
        # Sample X (vector of DIM dimensions)
        X = np.random.uniform(low=self.x_range[0], high=self.x_range[1], size=self.dim)

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


# if __name__ == '__main__':
    # np.random.seed = DataGenerator.RANDOM_STATE
    # gen = DataGenerator(poly_deg=3, size=500, dim=5, data_folder='./Data/')
    # gen()