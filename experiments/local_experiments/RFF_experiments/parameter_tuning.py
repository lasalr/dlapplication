import os
import sys
from datetime import datetime
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

sys.path.append("../../../../dlapplication")
sys.path.append("../../../../dlplatform")

from experiments.local_experiments.RFF_experiments.LinearSVCSampledRFF import LinearSVCSampledRFF
from experiments.local_experiments.RFF_experiments.data_handling import load_data, split_dataset, write_csv

RANDOM_STATE = 123

if __name__ == '__main__':
    start_time = datetime.now()
    file_path = '../../../data/SUSY/SUSY.csv'
    dim = 18  # SUSY has 18 features
    data_label_col = 0
    validation_file_path = os.path.join(os.path.dirname(file_path), 'split', 'VAL_' + os.path.basename(file_path))
    print('Splitting dataset...')
    split_dataset(file_path=file_path)  # Does not save if file is present
    X, y = load_data(path=validation_file_path, label_col=data_label_col, d=dim)
    print('Data loaded')

    # Parameter tuning for Linear SVC without RFF
    print('Starting: Parameter tuning for Linear SVC without RFF...')
    param_grid = {'C': [x for x in np.linspace(1, 10, 50)],
                  'dual': [True, False], 'random_state': [RANDOM_STATE]}

    gs_model = GridSearchCV(estimator=LinearSVC(), verbose=2, param_grid=param_grid, scoring='roc_auc', n_jobs=-1)
    gs_model.fit(X, y)
    write_csv(path='./Results/', name='param_tune_linearsvc_', start_time=start_time,
              results=gs_model.cv_results_, sortby_col='rank_test_score')

    # Parameter tuning for Linear SVC with RFF
    print('Starting: Parameter tuning for Linear SVC with RFF...')
    param_grid_rff = {'C': [x for x in np.linspace(1, 10, 50)],
                      'dual': (True, False), 'random_state': [RANDOM_STATE],
                      'rff_sampler_gamma': [x for x in np.linspace(0.00005, 0.005, 100)],
                      'rff_sampler_n_components': [29]}

    gs_model_rff = GridSearchCV(estimator=LinearSVCSampledRFF(), verbose=2, param_grid=param_grid_rff,
                                scoring='roc_auc', n_jobs=-1)
    gs_model_rff.fit(X, y)
    write_csv(path='./Results/', name='param_tune_linearsvc_rff_', start_time=start_time,
              results=gs_model_rff.cv_results_, sortby_col='rank_test_score')
