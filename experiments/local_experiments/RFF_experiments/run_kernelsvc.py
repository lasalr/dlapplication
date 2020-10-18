import os
import sys

from sklearn.model_selection import train_test_split
from itertools import product
from datetime import datetime

sys.path.append("../../../../dlapplication")
sys.path.append("../../../../dlplatform")

from experiments.local_experiments.RFF_experiments.data_handling import load_data, write_experiment
from experiments.local_experiments.RFF_experiments.training_evaluating import evaluate_model_roc_auc, train_rff_kernel_svm


RANDOM_STATE = 123


if __name__ == '__main__':
    start_time = datetime.now()
    dim = 18  # SUSY has 18 features
    reg_param = 0.01
    file_path = '../../../data/SUSY/SUSY.csv'

    X, y = load_data(path=file_path, label_col=0, d=dim)
    print('loaded data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    print('Split data')

    reg_param_initial = 0.01
    reg_params = [reg_param_initial]
    # reg_params = [reg_param_initial*5000, reg_param_initial*1000, reg_param_initial*500, reg_param_initial*100,
    #               reg_param_initial*10, reg_param_initial, reg_param_initial*0.30, reg_param_initial*0.10,
    #               reg_param_initial*0.05, reg_param_initial*0.01]

    kernel_gamma_vals = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    df_shapes = ['ovo', 'ovr']

    counter = 0
    total_count = len(list(product(reg_params, kernel_gamma_vals, df_shapes)))
    all_model_params = []
    for (reg_param, kernel_gamma, df_shape) in product(reg_params, kernel_gamma_vals, df_shapes):
        counter += 1
        print('\nExperiment number {} of {}, reg_param={}, kernel_gamma={}, df_shape={}'.
              format(counter, total_count, reg_param, kernel_gamma, df_shape))

        svc_model = train_rff_kernel_svm(X_train, y_train, c=reg_param, scale=True,
                                         svm_kernel_gamma=kernel_gamma, df_shape=df_shape)

        roc_auc = evaluate_model_roc_auc(model=svc_model, X_test=X_test, y_test=y_test)
        print('ROC AUC Score for {} model with={}'.format('Kernel SVM', roc_auc))

        # Appending all results and parameters into a list of dicts
        all_model_params.append({'experiment_no': counter, 'ROC_AUC': roc_auc, 'reg_param': reg_param,
                                 'kernel_gamma': kernel_gamma, 'df_shape': df_shape})

        # Sorting the list based on ROC_AUC

    # Writing results to file before ordering on ROC_AUC
    write_experiment(path='./Results/', name='find_rff_kernelsvc_sequenced_',
                     start_time=start_time, experiment_list=all_model_params)

    # Sorting results in descending order based on ROC_AUC
    all_model_params = sorted(all_model_params, key=lambda k: k['ROC_AUC'], reverse=True)

    # print('All the model parameters and results:\n{}'.format(all_model_params))
    print('Best model parameters:', all_model_params[0])

    # Writing results to file after ordering on ROC_AUC
    write_experiment(path=os.path.join(os.getcwd(), 'Results'), name='find_rff_kernelsvc_ranked_',
                     start_time=start_time, experiment_list=all_model_params)




