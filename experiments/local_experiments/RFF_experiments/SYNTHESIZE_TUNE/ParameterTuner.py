import os
import sys
from datetime import datetime

from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.append("../../../..")
sys.path.append("../../../../../dlplatform")

from experiments.local_experiments.RFF_experiments.data_handling import load_data, split_dataset, write_csv

class ParameterTuner:
    RANDOM_STATE = 123

    def __init__(self, C_list: list, rff_gamma_list: list, n_components_list: list, n_jobs: int, dataset_name,
                 val_file_path, results_folder_path, tune_data_fraction, dim, data_label_col, score_method='roc_auc'):
        self.C_list = C_list
        self.rff_gamma_list = rff_gamma_list
        self.n_components_list = n_components_list
        self.score_method = score_method
        self.n_jobs = n_jobs
        self.param_grid = {'C': self.C_list, 'dual': [False], 'random_state': [ParameterTuner.RANDOM_STATE]}
        self.param_grid_rff = {'svc__C': self.C_list, 'rff__gamma': self.rff_gamma_list,
                               'rff__random_state': [ParameterTuner.RANDOM_STATE],
                               'rff__n_components': self.n_components_list}
        self.results_folder_path = results_folder_path
        self.val_file_path = val_file_path
        self.dataset_name = dataset_name
        self.tune_data_fraction = tune_data_fraction
        self.dim = dim
        self.data_label_col = data_label_col

    def __call__(self):
        # Tune parameters
        tune_start_time = datetime.now()
        X, y = load_data(path=self.val_file_path, label_col=self.data_label_col, d=self.dim)

        # Taking fraction of data for tuning
        X_other, X_param, y_other, y_param = train_test_split(X, y, test_size=self.tune_data_fraction,
                                                              random_state=ParameterTuner.RANDOM_STATE)
        # Clear memory
        X_other = None
        y_other = None
        print('X_param={}\ny_param.shape={}'.format(X_param.shape, y_param.shape))
        print('Data loaded')

        # Parameter tuning for Linear SVC without RFF
        print('Starting: Parameter tuning for Linear SVC without RFF...')

        gs_model = GridSearchCV(estimator=LinearSVC(dual=False, max_iter=10000,
                                                    random_state=ParameterTuner.RANDOM_STATE),
                                verbose=1, param_grid=self.param_grid, scoring=self.score_method, n_jobs=self.n_jobs)
        gs_model.fit(X_param, y_param)
        print('writing results to file...')
        if not os.path.exists(self.results_folder_path):
            os.mkdir(self.results_folder_path)
        write_csv(path=self.results_folder_path, name='param_tune_linearsvc_' + self.dataset_name + '_',
                  start_time=tune_start_time, results=gs_model.cv_results_, sortby_col='rank_test_score')

        # Parameter tuning for Linear SVC with RFF
        print('Starting: Parameter tuning for Linear SVC with RFF...')
        pipe = Pipeline([('rff', RBFSampler()), ('svc', LinearSVC(dual=False, max_iter=10000,
                                                                  random_state=ParameterTuner.RANDOM_STATE))])

        gs_model_rff = GridSearchCV(estimator=pipe, verbose=1, param_grid=self.param_grid_rff,
                                    scoring=self.score_method, n_jobs=self.n_jobs)
        gs_model_rff.fit(X_param, y_param)
        print('writing results to file...')
        write_csv(path=self.results_folder_path, name='param_tune_linearsvc_rff_' + self.dataset_name,
                  start_time=tune_start_time, results=gs_model_rff.cv_results_, sortby_col='rank_test_score')

        return gs_model, gs_model_rff
