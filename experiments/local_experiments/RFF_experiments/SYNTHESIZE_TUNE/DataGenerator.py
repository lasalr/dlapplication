import functools
import itertools
import math
import operator
import os
import shutil
import statistics
import sys
import numpy as np
import scipy.special
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from experiments.local_experiments.RFF_experiments.data_handling import split_dataset

sys.path.append("../../../..")
sys.path.append("../../../../../dlplatform")


class DataGenerator:
    """
    Synthetic data generator
    """
    RANDOM_STATE = 123

    def __init__(self, poly_deg, size, dim, data_folder, data_name, method='custom', xy_noise_scale=None, x_range=None,
                 bias_range=None, min_max_coeff=None, gaussian_means=None, cov_matrix_random_state=RANDOM_STATE,
                 cov_matrix1=None, cov_matrix2=None):
        self.dim = dim
        self.poly_deg = poly_deg
        self.size = size
        self.method = method
        self.data_name = data_name

        if min_max_coeff is None:
            self.min_coef = -10
            self.max_coef = 10
        else:
            self.min_coef = min_max_coeff[0]
            self.max_coef = min_max_coeff[1]
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

        if gaussian_means is None:
            self.gaussian_means = [[5, 4, 7, -5, 0], [5.5, 3.5, 7.2, -5.1, 0.6]]
        else:
            self.gaussian_means = gaussian_means
        self.cov_matrix_random_state = cov_matrix_random_state

        if cov_matrix1 is None:
            self.cov_matrix1 = np.array([[10, 0, 1, 4, 1], [0, -5, 1, 5, 0], [5, 3, 0, 0, 1], [1, 1, 1, 1, 1], [4, 3, -2, 1, 0]])
        else:
            self.cov_matrix1 = cov_matrix1

        if cov_matrix2 is None:
            self.cov_matrix2 = np.array([[8, 1, 2, 3, 4], [1, -3, 2, 8, -1], [4, 2, -3, 5, 3], [1, 2, 2, 2, 1], [1, 7, -3, 2, -3]])
        else:
            self.cov_matrix2 = cov_matrix2

    def __call__(self):
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)

        # Copy script to results folder
        shutil.copy(__file__, os.path.join(self.data_folder, 'scripts', os.path.basename(__file__)))

        self.data_file_path = os.path.join(self.data_folder, self.data_name + '.csv')
        self.val_file_path = os.path.join(self.data_folder, os.path.dirname(self.data_file_path), 'split',
                                          'VAL_' + os.path.basename(self.data_file_path))
        self.train_data_path = os.path.join(self.data_folder, os.path.dirname(self.data_file_path), 'split',
                                            'TRAIN_' + os.path.basename(self.data_file_path))
        self.test_data_path = os.path.join(self.data_folder, os.path.dirname(self.data_file_path), 'split',
                                           'TEST_' + os.path.basename(self.data_file_path))

        np.random.seed = DataGenerator.RANDOM_STATE
        print("Generating data using method={}".format(self.method))
        if self.method == 'custom':
            coeffs = self.generate_coeffs()
            bias = np.random.uniform(low=self.bias_range[0], high=self.bias_range[1])
            size = self.size

            X, Y_values = [], []
            for i in range(size):
                x_point, y_point = self.generate_datapoint(coeffs, bias)
                X.append(x_point)
                Y_values.append(y_point)

            plt.plot(X)
            plt.savefig('figX1.png')

            # Scaling y
            scaler = StandardScaler()
            Y_values = np.array(Y_values)
            plt.plot(Y_values)
            plt.savefig('fig1.png')
            plt.hist(Y_values)
            plt.savefig('hist1.png')
            Y_values = scaler.fit_transform(Y_values.reshape(-1, 1))
            plt.plot(Y_values)
            plt.savefig('fig2.png')
            plt.hist(Y_values)
            plt.savefig('hist2.png')
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

        elif self.method == 'sklearn':
            X, Y = make_classification(n_samples=self.size, n_features=self.dim, n_informative=self.dim, n_redundant=0,
                                       n_classes=2, n_clusters_per_class=10, flip_y=self.xy_noise_scale[1], class_sep=1,
                                       scale=1, random_state=DataGenerator.RANDOM_STATE, hypercube=False)
            # Reduce class_sep to make harder
            # flip_y is label noise
            # hypercube If True, the clusters are put on the vertices of a hypercube. If False, the clusters are put on
            # the vertices of a random polytope.
            data = np.hstack((Y.reshape((Y.shape[0], 1)), X))
        elif self.method == 'gaussian':
            data = self.generate_gaussian_data()
        else:
            raise ValueError('Incorrect method given!')

        print('data.shape={}'.format(data.shape))

        np.savetxt(fname=self.data_file_path, X=data, comments='', delimiter=',')
        print('Splitting dataset...')
        split_dataset(file_path=os.path.abspath(self.data_file_path))  # Does not save if file is present
        return os.path.abspath(self.data_file_path)

    def generate_datapoint(self, coeffs, bias):
        # Sample X (vector of DIM dimensions)
        X = np.random.uniform(low=self.x_range[0], high=self.x_range[1], size=self.dim)
        # print(X)
        # Create polynomial combinations e.g. (x1^3, x2^3, x1^2 * x2, ...)
        #     (x1^2, x2^2, x3^2, x1*x2, ...) ... (...)
        poly = [itertools.combinations_with_replacement(X, r) for r in
                range(1, self.poly_deg + 1)]

        # Multiplying each tuple from above
        multiplied_polys_temp = [[functools.reduce(operator.mul, list(tup), 1) for tup in p] for
                                 p in poly]
        # print('multiplied_polys_temp={}'.format(multiplied_polys_temp))

        # Multiplying final values with coefficients
        multiplied_poly = [functools.reduce(operator.mul, val_list + [c], 1) for c, val_list in
                           zip(coeffs, multiplied_polys_temp)]

        # print('multiplied_poly={}'.format(multiplied_poly))
        # Calculating result
        y_val = bias + sum(multiplied_poly)
        # print('y_val={}'.format(y_val))
        # Return row of data
        return X, y_val

    def generate_gaussian_data(self):
        print('Generating Gaussian data')
        # For +1 class
        cov_m1 = self.cov_matrix1
        X1 = multivariate_normal.rvs(mean=self.gaussian_means[0], cov=cov_m1,
                                     size=int(math.ceil(self.size/2)), random_state=DataGenerator.RANDOM_STATE)
        print('X1.shape={}'.format(X1.shape))
        y1 = np.full(shape=(X1.shape[0], 1), fill_value=1, dtype=np.int16)
        print('y1.shape={}'.format(y1.shape))
        data1 = np.hstack((y1, X1))
        np.random.shuffle(data1)

        # For -1 class
        cov_m2 = self.cov_matrix2
        X2 = multivariate_normal.rvs(mean=self.gaussian_means[1], cov=cov_m2,
                                     size=int(math.floor(self.size/2)), random_state=DataGenerator.RANDOM_STATE)
        print('X2.shape={}'.format(X2.shape))
        y2 = np.full(shape=(X2.shape[0], 1), fill_value=-1, dtype=np.int16)
        print('y2.shape={}'.format(y2.shape))
        data2 = np.hstack((y2, X2))
        np.random.shuffle(data2)
        data = np.vstack((data1, data2))
        print('data.shape={}'.format(data.shape))
        return data

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