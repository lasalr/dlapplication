import multiprocessing
import os
import sys
import time

from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from itertools import product
from datetime import datetime

sys.path.append("../../../../dlapplication")
sys.path.append("../../../../dlplatform")

from experiments.local_experiments.RFF_experiments.data_handling import load_data, write_experiment
from experiments.local_experiments.RFF_experiments.training_evaluating import gamma_estimate, train_rff_linear_svc, \
    evaluate_model_roc_auc

RANDOM_STATE = 123
MAX_PROCESSES = 4


def train_eval_multiprocess(counter: int, total_count: int, X_train, y_train, X_test, y_test, g: float, n: int,
                            reg_param: float, return_queue: multiprocessing.Queue):
    print('\nRunning experiment {} of {}, n={}, g={}'.format(counter, total_count, n, g))
    rbf_sampler = RBFSampler(gamma=g, n_components=n, random_state=RANDOM_STATE)
    svc_model = train_rff_linear_svc(X_train, y_train, c=reg_param, sampler=rbf_sampler)
    roc_auc = evaluate_model_roc_auc(model=svc_model, X_test=X_test, y_test=y_test, sampler=rbf_sampler)
    print('Experiment {} of {}, ROC AUC Score for {} model with {} RFF components ={}'.
          format(counter, total_count, 'LinearSVC', n, roc_auc))

    # Adding dict of results and parameters to multiprocessing.Queue
    return_queue.put({'experiment_no': counter, 'ROC_AUC': roc_auc, 'rff_n_components': n, 'rff_gamma': g,
                      'reg_param': reg_param})


if __name__ == '__main__':
    start_time = datetime.now()
    dim = 18  # SUSY has 18 features
    reg_param = 0.01
    file_path = '../../../data/SUSY/SUSY.csv'

    X, y = load_data(path=file_path, label_col=0, d=dim)
    print('loaded data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    print('Split data')
    gamma_initial = gamma_estimate(X_train, 10000)
    print('gamma estimate={}\n'.format(gamma_initial))

    # Without RFF
    svc_model = train_rff_linear_svc(X_train, y_train, c=reg_param)
    print('ROC AUC Score for {} model without RFF={}'.
          format('LinearSVC', evaluate_model_roc_auc(model=svc_model, X_test=X_test, y_test=y_test)))

    gamma_initial = 0.005
    n_values = [x for x in range(15, 50, 2)]
    gamma_values = [gamma_initial]
    counter = 0
    total_count = len(list(product(n_values, gamma_values)))

    all_model_params = []
    jobs = []
    q = multiprocessing.Queue()
    for (n, g) in product(n_values, gamma_values):
        counter += 1
        p = multiprocessing.Process(target=train_eval_multiprocess,
                                    args=(counter, total_count, X_train, y_train, X_test, y_test, g, n, reg_param, q))
        jobs.append(p)
        p.start()
        print('Started process with ID={}'.format(str(os.getpid())))
        if len(jobs) > MAX_PROCESSES > q.qsize():
            time.sleep(5)

    for p in jobs:
        ret = q.get()
        all_model_params.append(ret)
        p.join()

        # Sorting the list based on ROC_AUC

    # Writing results to file before ordering on ROC_AUC
    write_experiment(path='./Results/', name='find_rff_linearsvc_sequenced_',
                     start_time=start_time, experiment_list=all_model_params)

    all_model_params = sorted(all_model_params, key=lambda k: k['ROC_AUC'], reverse=True)

    # print('All the model parameters and results:\n{}'.format(all_model_params))
    print('Best model parameters:', all_model_params[0])
    print('Worst model parameters:', all_model_params[-1])

    # Writing results to file after ordering on ROC_AUC
    write_experiment(path=os.path.join(os.getcwd(), 'Results'), name='find_rff_linearsvc_ranked_',
                     start_time=start_time, experiment_list=all_model_params)
