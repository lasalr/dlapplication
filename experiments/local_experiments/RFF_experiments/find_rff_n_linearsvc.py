from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from itertools import product

from experiments.local_experiments.RFF_experiments.data_handling import load_data
from experiments.local_experiments.RFF_experiments.training_evaluating import gamma_estimate, train_rff_linear_svc, \
    evaluate_model

RANDOM_STATE = 123


if __name__ == '__main__':
    dim = 18  # SUSY has 18 features
    reg_param = 0.01
    file_path = '../../../data/SUSY/SUSY.csv'

    X, y = load_data(path=file_path, label_col=0, d=dim)
    print('loaded data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    print('Split data')
    gamma_initial = gamma_estimate(X_train, 10000)
    print('gamma estimate={}'.format(gamma_initial))

    # Without RFF
    svc_model = train_rff_linear_svc(X_train, y_train, c=reg_param)
    print('ROC AUC Score for {} model without RFF={}'.
          format('LinearSVC', evaluate_model(X_test, y_test, model=svc_model)))

    gamma_initial = 0.1
    n_values = [300, 250, 200, 150, 100, 50, 25, 5]
    gamma_values = [gamma_initial, gamma_initial*0.90, gamma_initial*0.80, gamma_initial*0.60, gamma_initial*0.30,
                    gamma_initial*0.10, gamma_initial*0.05, gamma_initial*0.01]
    counter = 0
    total_count = len(list(product(n_values, gamma_values)))
    best_model_params = {'ROC_AUC': 0}
    for (n, g) in product(n_values, gamma_values):
        counter += 1
        print('Experiment number {} of {}, n={}, g={}'.format(counter, total_count, n, g))
        rbf_sampler = RBFSampler(gamma=g, n_components=n, random_state=RANDOM_STATE)
        svc_model = train_rff_linear_svc(X_train, y_train, c=reg_param, sampler=rbf_sampler)
        roc_auc = evaluate_model(X_test, y_test, model=svc_model, sampler=rbf_sampler)
        print('ROC AUC Score for {} model with {} RFF components ={}'.format('LinearSVC', n, roc_auc))

        if best_model_params['ROC_AUC'] < roc_auc:
            best_model_params['ROC_AUC'] = roc_auc
            best_model_params['n_components'] = n
            best_model_params['gamma'] = g
            best_model_params['experiment_no'] = counter

    print('best model parameters are', best_model_params)
