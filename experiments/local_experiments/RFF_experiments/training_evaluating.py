from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np

RANDOM_STATE = 123


def train_rff_linear_svc(X, y, c, sampler: RBFSampler = None):
    if sampler is not None:
        X = sampler.fit_transform(X)

    model = LinearSVC(C=c, max_iter=500, dual=False, random_state=RANDOM_STATE)
    model.fit(X, y)
    return model


def train_rff_kernel_svm(X, y, c):
    model = SVC(kernel='rbf', gamma='scale', C=c, max_iter=500, random_state=RANDOM_STATE)
    model.fit(X, y)
    return model


def gamma_estimate(features, n=100):
    index = np.random.choice(features.shape[0], n, replace=False)
    features = features[index]
    return pdist(features).mean()


def evaluate_model(X_test, y_test, model, sampler: RBFSampler = None):
    if sampler is not None:
        X_test_sampled = sampler.fit_transform(X_test)
        decision_scores = model.decision_function(X_test_sampled)
    else:
        decision_scores = model.decision_function(X_test)

    return roc_auc_score(y_true=y_test, y_score=decision_scores)