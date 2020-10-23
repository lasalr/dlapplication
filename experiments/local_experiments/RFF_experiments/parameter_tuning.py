import os
import sys
from datetime import datetime
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split

sys.path.append("../../../../dlapplication")
sys.path.append("../../../../dlplatform")

from experiments.local_experiments.RFF_experiments.data_handling import load_data, split_dataset, write_csv

RANDOM_STATE = 123

if __name__ == '__main__':
    start_time = datetime.now()
    file_path = '../../../data/SUSY/SUSY.csv'
    dim = 18  # SUSY has 18 features
    data_label_col = 0
    tune_data_fraction = 0.1
    validation_file_path = os.path.join(os.path.dirname(file_path), 'split', 'VAL_' + os.path.basename(file_path))
    print('Splitting dataset...')
    split_dataset(file_path=file_path)  # Does not save if file is present
    X, y = load_data(path=validation_file_path, label_col=data_label_col, d=dim)

    # Taking fraction of data for tuning
    X_other, X_param, y_other, y_param = train_test_split(X, y, test_size=tune_data_fraction, random_state=RANDOM_STATE)
    # Clear memory
    X_other = None
    y_other = None
    print('X_param={}\ny_param.shape={}'.format(X_param.shape, y_param.shape))
    print('Data loaded')

    # Parameter tuning for Linear SVC without RFF
    # print('Starting: Parameter tuning for Linear SVC without RFF...')
    # param_grid = {'C': [x for x in np.linspace(4, 7, 5)],
    #               'dual': [True, False], 'random_state': [RANDOM_STATE]}
    #
    # gs_model = GridSearchCV(estimator=LinearSVC(), verbose=1, param_grid=param_grid, scoring='roc_auc', n_jobs=-1)
    # gs_model.fit(X, y)
    # print('writing results to file...')
    # write_csv(path='./Results/', name='param_tune_linearsvc_', start_time=start_time,
    #           results=gs_model.cv_results_, sortby_col='rank_test_score')

    # Parameter tuning for Linear SVC with RFF
    print('Starting: Parameter tuning for Linear SVC with RFF...')

    param_grid_rff = {'svc__C': [2 ** x for x in range(0, 10)],
                      'svc__dual': [True, False], 'svc__random_state': [RANDOM_STATE],
                      'rff__gamma': [2 ** x for x in range(-14, 5)], 'rff__random_state': [RANDOM_STATE],
                      'rff__n_components': [x for x in range(2, 100, 5)]}

    pipe = Pipeline([('rff', RBFSampler()), ('svc', LinearSVC())])

    gs_model_rff = GridSearchCV(estimator=pipe, verbose=1, param_grid=param_grid_rff,
                                scoring='roc_auc', n_jobs=-1)
    gs_model_rff.fit(X_param, y_param)
    print('writing results to file...')
    write_csv(path='./Results/', name='param_tune_linearsvc_rff_', start_time=start_time,
              results=gs_model_rff.cv_results_, sortby_col='rank_test_score')
