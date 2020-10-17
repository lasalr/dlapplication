from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import pdist
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np

RANDOM_STATE = 123


def train_rff_linear_svc(X, y, c, sampler: RBFSampler = None, scale=False):
    model = LinearSVC(C=c, max_iter=500, dual=False, random_state=RANDOM_STATE)

    if sampler is not None:
        X = sampler.fit_transform(X)

    if scale:
        model_pipeline = make_pipeline(StandardScaler(), model)
        model_pipeline.fit(X, y)
        return model_pipeline
    else:
        model.fit(X, y)
        return model


def train_rff_kernel_svm(X, y, c, scale=False):
    model = SVC(kernel='rbf', gamma='auto', C=c, max_iter=1000, random_state=RANDOM_STATE)

    if scale:
        model_pipeline = make_pipeline(StandardScaler(), model)
        model_pipeline.fit(X, y)
        return model_pipeline
    else:
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