import os

from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import RBFSampler


class LinearSVCSampledRFF(LinearSVC):
    def __init__(self, penalty='l2', loss='squared_hinge', rff_sampler_gamma=1, rff_sampler_n_components=100,
                 dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                 class_weight=None, verbose=0, random_state=None, max_iter=1000):

        super(LinearSVCSampledRFF, self).__init__(penalty=penalty, loss=loss, dual=dual, tol=tol,
                                                  C=C, multi_class=multi_class, fit_intercept=fit_intercept,
                                                  intercept_scaling=intercept_scaling, class_weight=class_weight,
                                                  verbose=verbose, random_state=random_state, max_iter=max_iter)

        self.rff_sampler_gamma = rff_sampler_gamma
        self.rff_sampler_n_components = rff_sampler_n_components
        self.sampler = RBFSampler(gamma=self.rff_sampler_gamma, n_components=self.rff_sampler_n_components,
                                  random_state=self.random_state)

        # print('rff_sampler_gamma=', self.rff_sampler_gamma)
        # print('rff_sampler_n_components=', self.rff_sampler_n_components)
        # print('dual=', self.dual)
        # print('C=', self.C)

    def fit(self, X, y, sample_weight=None):
        if self.sampler is not None:
            X = self.sampler.fit_transform(X)
            # print('RBF transform done before fit() with pid{} X.dtype={}'.format(os.getpid(), X.dtype))

        super(LinearSVCSampledRFF, self).fit(X, y, sample_weight)

    def predict(self, X):
        if self.sampler is not None:
            X = self.sampler.fit_transform(X)

        super(LinearSVCSampledRFF, self).predict(X)

    def score(self, X, y, sample_weight=None):
        if self.sampler is not None:
            X = self.sampler.fit_transform(X)

        super(LinearSVCSampledRFF, self).score(X, y, sample_weight)

    def decision_function(self, X):
        if self.sampler is not None:
            X = self.sampler.fit_transform(X)

        super(LinearSVCSampledRFF, self).decision_function(X)
