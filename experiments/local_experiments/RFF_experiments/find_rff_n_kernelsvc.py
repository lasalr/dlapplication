from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from itertools import product

from experiments.local_experiments.RFF_experiments.data_handling import load_data
from experiments.local_experiments.RFF_experiments.training_evaluating import gamma_estimate, train_rff_linear_svc, \
    evaluate_model, train_rff_kernel_svm

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

    # Without RFF Linear SVC
    svc_model = train_rff_linear_svc(X_train, y_train, c=reg_param)
    print('ROC AUC Score for {} model without RFF={}'.
          format('LinearSVC', evaluate_model(X_test, y_test, model=svc_model)))

    # Without RFF kernel SVC
    kernel_type = 'rbf'
    svc_model = train_rff_kernel_svm(X_train, y_train, c=reg_param)
    print('ROC AUC Score for {} model without RFF and {} kernel={}'.
          format('SVC (kernel)', kernel_type, evaluate_model(X_test, y_test, model=svc_model)))



