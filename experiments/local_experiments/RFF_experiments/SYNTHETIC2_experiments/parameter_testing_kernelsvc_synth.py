import os
import shutil
import sys
from datetime import datetime
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

sys.path.append("../../../..")
sys.path.append("../../../../../dlplatform")

from experiments.local_experiments.RFF_experiments.data_handling import load_data, split_dataset, write_csv, \
    create_get_ts_folder

RANDOM_STATE = 123

if __name__ == '__main__':
    start_time = datetime.now()
    file_path = '../../../../data/SYNTHETIC2/SYNTHETIC_DATA.csv'
    ds_name = 'SYNTHETIC2'
    dim = 5  # SYNTHETIC has 5 features
    data_label_col = 0
    keep_fraction = 0.3
    validation_file_path = os.path.join(os.path.dirname(file_path), 'split', 'VAL_' + os.path.basename(file_path))

    print('Splitting dataset...')
    split_dataset(file_path=file_path)  # Does not save if file is present
    X, y = load_data(path=validation_file_path, label_col=data_label_col, d=dim)

    # Taking fraction of data for tuning
    X_other, X_keep, y_other, y_keep = train_test_split(X, y, test_size=keep_fraction, random_state=RANDOM_STATE)
    # Clear memory
    X_other = None
    y_other = None
    print('X_keep={}\ny_keep.shape={}'.format(X_keep.shape, y_keep.shape))
    print('Data loaded')

    X_train, X_test, y_train, y_test = train_test_split(X_keep, y_keep, test_size=0.3, random_state=RANDOM_STATE)
    print('X_train={}\ny_train.shape={}'.format(X_train.shape, y_train.shape))
    # Linear SVC without RFF
    k_svc = 8192
    k_svc_gamma = 0.0078125
    model = SVC(C=k_svc, gamma=k_svc_gamma, decision_function_shape='ovo', random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_decisions = model.decision_function(X_test)

    print('writing results to file...')

    out_file = os.path.join('./Results/', 'Results_' + 'param_test_kernelsvc_' +
                            str(start_time).replace(':', '_').replace(' ', '_') + '.txt')

    out_string = 'Start time: {}\n'.format(str(start_time))
    out_string += 'End time: {}\n'.format(str(datetime.now()))
    out_string += 'Parameters:\n{}\n'.format("SVC(C={}, gamma={}, decision_function_shape='ovo', random_state={})".format(k_svc, k_svc_gamma, RANDOM_STATE))
    out_string += 'Accuracy={}\nROCAUC={}'.format(accuracy_score(y_true=y_test, y_pred=y_pred),
                                                  roc_auc_score(y_true=y_test, y_score=y_decisions))
    print('Kernel SVC:\n', out_string)

    with open(out_file, 'w') as f:
        f.write(out_string)


