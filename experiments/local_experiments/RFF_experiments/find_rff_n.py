from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist
from sklearn.kernel_approximation import RBFSampler
from itertools import product

RANDOM_STATE = 123


def load_data(path: str, label_col: int, d: int):
    df = pd.read_csv(filepath_or_buffer=path, names=[x for x in range(0, d + 1)])
    labels = df.iloc[:, label_col].to_numpy(dtype=np.int32)
    features = df.drop(df.columns[label_col], axis=1).to_numpy()
    return features, labels


def train_rff_linear_svc(X, y, c, sampler: RBFSampler = None):
    if sampler is not None:
        X = sampler.fit_transform(X)

    model = LinearSVC(C=c, max_iter=500, dual=False, random_state=RANDOM_STATE)
    model.fit(X, y)
    return model


def gamma_estimate(features, n=100):
    index = np.random.choice(features.shape[0], n, replace=False)
    features = features[index]
    return pdist(features).mean()


def evaluate_model(X_test, y_test, model: LinearSVC, sampler: RBFSampler = None):
    if sampler is not None:
        X_test_sampled = sampler.fit_transform(X_test)
        decision_scores = model.decision_function(X_test_sampled)
    else:
        decision_scores = model.decision_function(X_test)

    return roc_auc_score(y_true=y_test, y_score=decision_scores)


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

    gamma_initial = 0.3
    n_values = [100, 50, 25]
    gamma_values = [gamma_initial, gamma_initial*0.99, gamma_initial*0.95, gamma_initial*0.90, gamma_initial*0.80,
                    gamma_initial*0.60, gamma_initial*0.30, gamma_initial*0.10, gamma_initial*0.05, gamma_initial*0.01]
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
