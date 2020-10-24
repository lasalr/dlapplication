import os
import shutil
import sys
from datetime import datetime
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.append("../../../../dlapplication")
sys.path.append("../../../../dlplatform")

from experiments.local_experiments.RFF_experiments.LinearSVCSampledRFF import LinearSVCSampledRFF
from experiments.local_experiments.RFF_experiments.data_handling import load_data, split_dataset, write_csv, \
    create_get_ts_folder

RANDOM_STATE = 123

if __name__ == '__main__':
    start_time = datetime.now()

    file_path = '../../../data/SUSY/SUSY.csv'
    dim = 18  # SUSY has 18 features
    data_label_col = 0
    test_fraction = 0.2
    validation_file_path = os.path.join(os.path.dirname(file_path), 'split', 'VAL_' + os.path.basename(file_path))

    # Creating timestamp folder
    print('Creating timestamp folder...')
    ts_folder = create_get_ts_folder(script_file_path=os.path.realpath(__file__), start_time=start_time)

    print('Splitting dataset...')
    split_dataset(file_path=file_path)  # Does not save if file is present
    X, y = load_data(path=validation_file_path, label_col=data_label_col, d=dim)

    # Splitting train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=RANDOM_STATE)

    print('Data loaded')
    print('X_train={}\ny_train.shape={}\nX_test={}\ny_test.shape={}'.
          format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=RANDOM_STATE, C=257.5,
                                               decision_function_shape='ovo', gamma=0.001953125))
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_decisions = pipe.decision_function(X_test)

    out_string = 'Start time: {}\n'.format(str(start_time))
    out_string += 'End time: {}\n'.format(str(datetime.now()))
    out_string += 'Accuracy={}\nROCAUC={}'.format(accuracy_score(y_true=y_test, y_pred=y_pred),
                                                  roc_auc_score(y_true=y_test, y_score=y_decisions))
    print(out_string)

    # Writing results to file in timestamped folder
    with open(os.path.join(ts_folder, 'results.txt'), 'w') as f:
        f.write(out_string)

    # Copying python script to new folder with timestamp
    ts = os.path.basename(os.path.normpath(ts_folder))

    # Saving timestamped copy of current script file
    shutil.copy(src=__file__, dst=os.path.join(ts_folder, os.path.basename(__file__) + ts))
