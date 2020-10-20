import os
import sys
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append("../../../../dlapplication")
sys.path.append("../../../../dlplatform")

from experiments.local_experiments.RFF_experiments.data_handling import load_data, split_dataset, write_csv

RANDOM_STATE = 123

if __name__ == '__main__':
    start_time = datetime.now()
    dim = 18  # SUSY has 18 features
    reg_param = 0.01
    file_path = '../../../data/SUSY/SUSY.csv'
    data_label_col = 0
    validation_file_path = os.path.join(os.path.dirname(file_path), 'split', 'VAL_' + os.path.basename(file_path))

    print('Splitting dataset...')
    split_dataset(file_path=file_path)  # Does not save if file is present
    X, y = load_data(path=validation_file_path, label_col=data_label_col, d=dim)
    print('Data loaded')

    pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

    reg_param_initial = 0.01
    gamma_initial = 0.0001
    param_grid = {
        'svc__C': [reg_param_initial * 5000, reg_param_initial * 1000, reg_param_initial * 500, reg_param_initial * 100,
                   reg_param_initial * 10, reg_param_initial, reg_param_initial * 0.30, reg_param_initial * 0.10,
                   reg_param_initial * 0.05, reg_param_initial * 0.01],
        'svc__gamma': ['scale', 'auto', gamma_initial * 5000, gamma_initial * 1000, gamma_initial * 500,
                       gamma_initial * 100, gamma_initial * 10, gamma_initial, gamma_initial * 0.30,
                       gamma_initial * 0.10, gamma_initial * 0.05, gamma_initial * 0.01],
        'svc__random_state': [RANDOM_STATE],
        'svc__decision_function_shape': ['ovo']}

    gs_model_kernelsvm = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, verbose=1, scoring='roc_auc',
                                      n_jobs=-1).fit(X, y)
    gs_model_kernelsvm.fit(X)

    write_csv(path='./Results/', name='param_tune_kernelsvc_', start_time=start_time,
              results=gs_model_kernelsvm.cv_results_, sortby_col='rank_test_score')
